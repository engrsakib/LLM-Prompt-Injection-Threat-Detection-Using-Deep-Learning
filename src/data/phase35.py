"""Phase 3.5 - leakage-aware, group-stratified re-split of the LOCAL raw snapshot.

Pipeline (raw files are never modified):

1. read ``data/raw/{train,validation,test}.jsonl`` (offline; no Hugging Face access),
2. clean + validate (same rules as Phase 1; severity range 1-3),
3. build duplicate / template / near-duplicate families (``src.data.families``),
4. assign whole families to train / validation / test with stratified greedy placement
   (``src.data.resplit``), preferring the upstream split as tie-break,
5. within-split exact (+ optional near) de-duplication,
6. obfuscation-aware columns (same as Phase 2) so the training text column is consistent,
7. full leakage audit (raw / normalised / near / family) - must pass,
8. write versioned splits + manifest + dropped-row log + hashes,
9. regenerate train-only variants (balanced / augmented / adversarial) from the NEW train
   split and audit them against the new validation / test,
10. export ambiguity + technique subsets and a Kaggle package for the version.

Nothing here trains a model; ``manifest.json`` with ``frozen: true`` is the gate consumed
by ``src.eval`` before any production metric is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data.adversarial_paraphrase import AdversarialParaphraseConfig, augment_adversarial_paraphrase
from src.data.analysis import add_derived_columns, write_phase2_reports
from src.data.augment import AugmentConfig, augment_training_frame
from src.data.balancing import balance_minority_classes
from src.data.cleaning import CleaningConfig, clean_text, cleaning_fingerprint
from src.data.constants import INTENT_LABELS, SPLITS
from src.data.families import DEFAULT_TEMPLATE_PATTERN, build_families, minhash_for
from src.data.kaggle_package import export_technique_subsets, package_kaggle_dataset
from src.data.leakage import LeakageError, assert_no_leakage, run_leakage_audit, write_leakage_report
from src.data.obfuscation import obfuscation_features
from src.data.reporting import write_distribution_reports
from src.data.resplit import DEFAULT_FRACTIONS, group_stratified_split
from src.data.validation import validate_record
from src.utils.io import iter_jsonl, save_jsonl, sha256_file, write_json

logger = logging.getLogger(__name__)

TRAIN_VARIANT_FILES = ("train_balanced.parquet", "train_augmented.parquet", "train_adversarial.parquet")


@dataclass
class Phase35Config:
    seed: int = 42
    data_dir: Path = Path("data")
    version: str = "v1_leakfree"
    fractions: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    near_threshold: float = 0.85
    within_split_dedup: str = "near"  # exact | near | none
    template_pattern: str = DEFAULT_TEMPLATE_PATTERN
    tolerance: float = 0.02
    lowercase: bool = False
    regenerate_train_variants: bool = True
    balancing_strategy: str = "oversample_median"
    augmentation: dict = field(default_factory=dict)
    adversarial_max_variants: int = 1
    build_kaggle_package: bool = True
    allow_leakage_failure: bool = False

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "processed" / "splits" / self.version

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase35Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        data_cfg = payload.get("data", {})
        p35 = payload.get("phase35", {})
        p2 = payload.get("phase2", {})
        p3 = payload.get("phase3", {})
        return cls(
            seed=int(payload.get("seed", 42)),
            data_dir=Path(data_cfg.get("data_dir", "data")),
            version=str(p35.get("version", "v1_leakfree")),
            fractions=dict(p35.get("fractions", DEFAULT_FRACTIONS)),
            near_threshold=float(p35.get("near_threshold", p3.get("leakage_near_threshold", 0.85))),
            within_split_dedup=str(p35.get("within_split_dedup", "near")),
            template_pattern=str(p35.get("template_pattern", DEFAULT_TEMPLATE_PATTERN)),
            tolerance=float(p35.get("tolerance", 0.02)),
            lowercase=bool(data_cfg.get("lowercase", False)),
            regenerate_train_variants=bool(p35.get("regenerate_train_variants", True)),
            balancing_strategy=str(p2.get("balancing_strategy", "oversample_median")),
            augmentation=dict(p2.get("augmentation", {})),
            adversarial_max_variants=int(p3.get("adversarial_paraphrase", {}).get("max_variants_per_row", 1)),
            build_kaggle_package=bool(p35.get("build_kaggle_package", True)),
            allow_leakage_failure=bool(p35.get("allow_leakage_failure", False)),
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def load_raw_splits(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = raw_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw split: {path} (raw data must exist locally)")
        rows = list(iter_jsonl(path))
        frame = pd.DataFrame(rows)
        frame["upstream_split"] = split
        frame["upstream_index"] = range(len(frame))
        frame["row_id"] = [f"{split}:{i}" for i in range(len(frame))]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if "intent_label" not in df.columns and "label" in df.columns:
        df["intent_label"] = df["label"]
    if "label" not in df.columns:
        df["label"] = df["intent_label"]
    return df


def clean_and_validate(df: pd.DataFrame, lowercase: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    cleaning = CleaningConfig(lowercase=lowercase)
    df = df.copy()
    df["text_original"] = df["text"].astype(str)
    df["text_clean"] = [clean_text(t, cleaning) for t in df["text_original"]]
    df["text"] = df["text_clean"]
    df["text_hash"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in df["text_original"]]
    df["text_fingerprint"] = [
        hashlib.sha256(cleaning_fingerprint(t).encode("utf-8")).hexdigest() for t in df["text_clean"]
    ]
    dropped: list[dict] = []
    keep_mask = []
    for pos, record in enumerate(df.to_dict(orient="records")):
        issues = [i for i in validate_record(record, pos) if i.severity == "error"]
        if issues:
            dropped.append(
                {
                    "row_id": record["row_id"],
                    "reason": "validation_error",
                    "details": [f"{i.field}: {i.message}" for i in issues],
                    "upstream_split": record["upstream_split"],
                    "intent_label": record.get("intent_label"),
                }
            )
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    return df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True), dropped


def within_split_dedup(
    df: pd.DataFrame,
    mode: str,
    threshold: float,
) -> tuple[pd.DataFrame, list[dict], dict[str, dict[str, int]]]:
    """Exact (+ optional near) de-duplication *inside* each split. Families stay intact."""
    if mode not in ("exact", "near", "none"):
        raise ValueError("within_split_dedup must be exact|near|none")
    dropped: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    kept_parts = []
    for split in SPLITS:
        part = df[df["split"] == split].sort_values(["upstream_split", "upstream_index"]).copy()
        before = len(part)
        exact_removed = near_removed = 0
        if mode in ("exact", "near"):
            seen: set[str] = set()
            keep = []
            for fp, rid in zip(part["text_fingerprint"], part["row_id"]):
                if fp in seen:
                    keep.append(False)
                    exact_removed += 1
                    dropped.append({"row_id": rid, "reason": "exact_duplicate_within_split", "split": split})
                else:
                    seen.add(fp)
                    keep.append(True)
            part = part[keep]
        if mode == "near":
            from datasketch import MinHashLSH

            try:
                lsh = MinHashLSH(threshold=threshold, num_perm=128)
            except ValueError:
                lsh = None
            if lsh is not None:
                keep = []
                for idx, (text, rid) in enumerate(zip(part["text_clean"], part["row_id"])):
                    mh = minhash_for(text)
                    if mh is None:
                        keep.append(True)
                        continue
                    if lsh.query(mh):
                        keep.append(False)
                        near_removed += 1
                        dropped.append({"row_id": rid, "reason": "near_duplicate_within_split", "split": split})
                    else:
                        lsh.insert(f"{split}-{idx}", mh)
                        keep.append(True)
                part = part[keep]
        stats[split] = {"before": before, "exact_removed": exact_removed, "near_removed": near_removed, "after": len(part)}
        kept_parts.append(part)
    return pd.concat(kept_parts, ignore_index=True), dropped, stats


def add_obfuscation_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in df.to_dict(orient="records"):
        features = obfuscation_features(record["text_clean"], intent_label=int(record["intent_label"]))
        record.update(features)
        record["text"] = record["text_obfuscation_aware"]
        rows.append(record)
    return add_derived_columns(pd.DataFrame(rows))


def _class_table(df: pd.DataFrame, col: str = "split") -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for split, part in df.groupby(col):
        table[str(split)] = {
            INTENT_LABELS[int(k)]: int(v) for k, v in sorted(part["intent_label"].astype(int).value_counts().items())
        }
        table[str(split)]["_total"] = int(len(part))
    return table


def _source_table(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "source" not in df.columns:
        return {}
    out: dict[str, dict[str, int]] = {}
    for split, part in df.groupby("split"):
        counts = part["source"].astype(str).value_counts().to_dict()
        synthetic = int(sum(v for k, v in counts.items() if k.startswith("synthetic")))
        out[str(split)] = {**{str(k): int(v) for k, v in counts.items()}, "_synthetic_total": synthetic, "_total": int(len(part))}
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_phase35(config: Phase35Config) -> dict:
    raw_dir = config.data_dir / "raw"
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    snapshot_meta_path = raw_dir / "snapshot_metadata.json"
    snapshot_meta = json.loads(snapshot_meta_path.read_text(encoding="utf-8")) if snapshot_meta_path.exists() else {}

    # 1) raw
    logger.info("Loading raw splits from %s", raw_dir)
    raw = load_raw_splits(raw_dir)
    raw_counts = raw["upstream_split"].value_counts().to_dict()
    logger.info("Raw rows: %s", raw_counts)

    # 2) clean + validate
    cleaned, dropped = clean_and_validate(raw, lowercase=config.lowercase)
    logger.info("After validation: %d rows (%d dropped)", len(cleaned), len(dropped))

    # 3) families
    family_ids, _fps, family_report = build_families(
        cleaned["text_clean"].tolist(),
        near_threshold=config.near_threshold,
        template_pattern=config.template_pattern,
    )
    cleaned["family_id"] = family_ids
    logger.info("Families: %d (largest %d)", family_report.families, family_report.largest_family)

    # 4) group-aware split
    assigned, resplit_report = group_stratified_split(
        cleaned,
        fractions=config.fractions,
        tolerance=config.tolerance,
    )
    cleaned["split"] = assigned.values
    logger.info("Re-split: kept %d rows in upstream split, moved %d", resplit_report.rows_kept_in_upstream_split, resplit_report.rows_moved)

    # 5) within-split dedup
    deduped, dedup_dropped, dedup_stats = within_split_dedup(cleaned, config.within_split_dedup, config.near_threshold)
    dropped.extend(dedup_dropped)

    # 6) obfuscation-aware columns
    final = add_obfuscation_columns(deduped)

    # 7) leakage audit (full) on the new splits
    split_frames = {s: final[final["split"] == s].reset_index(drop=True) for s in SPLITS}
    leakage = run_leakage_audit(split_frames, near_threshold=config.near_threshold, full=True)
    write_leakage_report(leakage, reports_dir / "leakage_report.json")
    logger.info("Leakage audit passed: %s", leakage["passed"])
    if not leakage["passed"] and not config.allow_leakage_failure:
        raise LeakageError(f"Phase 3.5 leakage audit failed: {json.dumps(leakage['summary'])}")

    # 8) write splits + logs
    hashes: dict[str, str] = {}
    for split, frame in split_frames.items():
        path = out_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        hashes[path.name] = sha256_file(path)
    save_jsonl(dropped, out_dir / "dropped_rows.jsonl")
    final[["row_id", "upstream_split", "split", "family_id", "intent_label", "text_hash"]].to_parquet(
        out_dir / "family_assignments.parquet", index=False
    )
    write_distribution_reports(split_frames, reports_dir)

    # 9) train-only variants
    variant_reports: dict = {}
    variant_leakage: dict = {}
    if config.regenerate_train_variants:
        train_df = split_frames["train"].copy()
        train_df["parent_row_id"] = train_df["row_id"]
        balanced, bal_report = balance_minority_classes(train_df, strategy=config.balancing_strategy, seed=config.seed)
        balanced.to_parquet(out_dir / "train_balanced.parquet", index=False)
        aug_cfg = AugmentConfig(
            enabled=bool(config.augmentation.get("enabled", True)),
            synonym_replace_prob=float(config.augmentation.get("synonym_replace_prob", 0.25)),
            token_noise_prob=float(config.augmentation.get("token_noise_prob", 0.10)),
            max_augmentations_per_row=int(config.augmentation.get("max_augmentations_per_row", 1)),
            augment_minority_only=bool(config.augmentation.get("minority_only", True)),
        )
        augmented, aug_report = augment_training_frame(balanced, config=aug_cfg, seed=config.seed)
        augmented.to_parquet(out_dir / "train_augmented.parquet", index=False)
        adversarial, adv_report = augment_adversarial_paraphrase(
            augmented,
            config=AdversarialParaphraseConfig(enabled=True, max_variants_per_row=config.adversarial_max_variants),
            seed=config.seed,
        )
        adversarial.to_parquet(out_dir / "train_adversarial.parquet", index=False)
        for name in TRAIN_VARIANT_FILES:
            hashes[name] = sha256_file(out_dir / name)
        variant_reports = {"balancing": bal_report, "augmentation": aug_report, "adversarial_paraphrase": adv_report}

        # Every variant row must descend from a train parent
        for name, frame in (("train_balanced", balanced), ("train_augmented", augmented), ("train_adversarial", adversarial)):
            parents = set(frame["parent_row_id"].astype(str))
            assert parents <= set(train_df["row_id"].astype(str)), f"{name}: variant rows with non-train parents"
            audit = run_leakage_audit(
                {"train": frame, "validation": split_frames["validation"], "test": split_frames["test"]},
                near_threshold=config.near_threshold,
                full=True,
                include_raw=False,
            )
            variant_leakage[name] = audit["summary"] | {"passed": audit["passed"]}
            write_leakage_report(audit, reports_dir / f"leakage_report_{name}.json")

    # 10) subsets + kaggle package
    subsets_dir = out_dir / "subsets"
    phase2_reports = write_phase2_reports(split_frames, reports_dir, subsets_dir)
    technique_report = export_technique_subsets(split_frames, subsets_dir)

    kaggle_manifest: dict = {}
    if config.build_kaggle_package:
        files = tuple(["train.parquet", "validation.parquet", "test.parquet", "manifest.json", *TRAIN_VARIANT_FILES])
        # manifest.json is written below; write a provisional copy first so it is included.
        _write_manifest(out_dir, config, snapshot_meta, raw_counts, dropped, family_report, resplit_report, dedup_stats,
                        final, leakage, variant_reports, variant_leakage, hashes, phase2_reports, technique_report, {}, frozen=False)
        kaggle_manifest = package_kaggle_dataset(
            out_dir,
            output_dir=out_dir / "kaggle_package",
            files=files,
            zip_path=out_dir / f"kaggle_package_{config.version}.zip",
            title=f"Prompt Injection Threat Matrix - {config.version} (leakage-aware re-split)",
            slug=f"prompt-injection-threat-matrix-{config.version.replace('_', '-')}",
            extra_readme=(
                "\n`train_*` variants are generated from the new train split only. Validation and test are "
                "untouched held-out splits; see `manifest.json` for hashes and the leakage audit.\n"
            ),
        )

    manifest = _write_manifest(out_dir, config, snapshot_meta, raw_counts, dropped, family_report, resplit_report, dedup_stats,
                               final, leakage, variant_reports, variant_leakage, hashes, phase2_reports, technique_report,
                               kaggle_manifest, frozen=True)
    return manifest


def _write_manifest(out_dir, config, snapshot_meta, raw_counts, dropped, family_report, resplit_report, dedup_stats,
                    final, leakage, variant_reports, variant_leakage, hashes, phase2_reports, technique_report,
                    kaggle_manifest, frozen: bool) -> dict:
    drop_reasons: dict[str, int] = {}
    for item in dropped:
        drop_reasons[item["reason"]] = drop_reasons.get(item["reason"], 0) + 1
    manifest = {
        "version": config.version,
        "frozen": frozen,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "raw_dir": str((config.data_dir / "raw").as_posix()),
            "snapshot_metadata": snapshot_meta,
            "raw_counts": {k: int(v) for k, v in raw_counts.items()},
            "provenance_caveat": (
                "Origin / licence are taken from local snapshot_metadata.json; the upstream page could not be "
                "re-verified at build time."
            ),
        },
        "config": {
            "seed": config.seed,
            "fractions": config.fractions,
            "near_threshold": config.near_threshold,
            "within_split_dedup": config.within_split_dedup,
            "template_pattern": config.template_pattern,
            "tolerance": config.tolerance,
            "lowercase": config.lowercase,
        },
        "families": family_report.to_dict(),
        "resplit": resplit_report.to_dict(),
        "within_split_dedup": dedup_stats,
        "dropped_rows": {"total": len(dropped), "by_reason": drop_reasons, "log": "dropped_rows.jsonl"},
        "final_counts": _class_table(final),
        "source_counts": _source_table(final),
        "leakage_audit": leakage,
        "leakage_passed": bool(leakage["passed"]),
        "train_variants": variant_reports,
        "train_variant_leakage": variant_leakage,
        "subsets": {"phase2": phase2_reports, "technique": technique_report},
        "kaggle_package": kaggle_manifest,
        "artifact_hashes_sha256": hashes,
        "caveat": (
            "Passing heuristic duplicate/near-duplicate/family checks does not prove zero statistical leakage."
        ),
    }
    write_json(manifest, out_dir / "manifest.json")
    return manifest


def load_frozen_manifest(splits_dir: str | Path) -> dict:
    """Load and validate the manifest of a versioned split directory (used as an evaluation gate)."""
    path = Path(splits_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json in {splits_dir}; run `python -m src.data.phase35` first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("frozen"):
        raise LeakageError(f"Split manifest {path} is not frozen")
    assert_no_leakage(manifest.get("leakage_audit", {}), context=f"for {splits_dir}")
    for name, digest in manifest.get("artifact_hashes_sha256", {}).items():
        file = Path(splits_dir) / name
        if file.exists() and sha256_file(file) != digest:
            raise LeakageError(f"Artifact {file} changed since the manifest was frozen")
    return manifest


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.5: leakage-aware group re-split of the local raw snapshot.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--version", default=None, help="Override phase35.version")
    parser.add_argument("--allow-leakage-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config = Phase35Config.from_yaml(args.config)
    if args.version:
        config.version = args.version
    if args.allow_leakage_failure:
        config.allow_leakage_failure = True
    manifest = run_phase35(config)
    print(f"Phase 3.5 completed -> {config.output_dir}")
    print(f"Leakage passed: {manifest['leakage_passed']}")
    for split, counts in manifest["final_counts"].items():
        print(f"  {split}: {counts['_total']} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
