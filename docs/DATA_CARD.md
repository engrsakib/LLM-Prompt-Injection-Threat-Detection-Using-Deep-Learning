# Data Card — Prompt Injection Threat Matrix (Processed)

## Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | Neuralchemy Prompt Injection Threat Matrix (Processed) |
| **Source** | [neuralchemy/prompt-injection-Threat-Matrix](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix) |
| **License** | CC BY-NC 4.0 (upstream dataset) |
| **Task** | Text classification (binary + 7-class intent) |
| **Language** | English |
| **Authors (project)** | Md. Nazmus Sakib, Kazi Omar Faruq |
| **Institution** | Daffodil International University |

## Source Dataset

The local snapshot (`data/raw/`, id `tm-4e7fede7adba`) contains **32,320** rows (25,856 / 3,232 / 3,232 in the
upstream train / validation / test files). Measured facts on this snapshot - they override the upstream description
where they differ:

- 7 intent classes; `label == intent_label` and `binary_label == (intent_label != 0)` for **every** row
- `severity` takes only the values **{1, 2, 3}** (upstream docs say 1-10; this snapshot does not contain 4-10)
- `surface` is the constant `user_input` and carries no information
- `source` categories: `hackaprompt`, `core`, `synthetic_threat`, `synthetic_benign`, `synthetic_short`;
  synthetic share 37.9% / 37.5% / 38.6% per raw split. Synthetic rows are kept; results are reported for
  synthetic vs non-synthetic subsets separately
- 200 benign prompts are templates with an `(example N)` suffix; 198 of those template families span more than
  one upstream split (4,042 / 490 / 516 rows), and the raw files share 14 / 20 / 4 exact duplicates between
  train-validation / train-test / validation-test
- Ambiguity flag for borderline cases; technique metadata for attacks

**Provenance / licence caveat.** The licence (CC BY-NC 4.0) and origin are taken from `snapshot_metadata.json`;
the upstream Hugging Face page could not be re-verified at build time. Do not upload or redistribute the data
without verified permission. The pipeline never re-downloads: all processing starts from the local raw files.

Upstream splits are **not** used for model evaluation because of the overlaps above - see *Phase 3.5* below.

## Processing Pipeline (Phase 1)

Pipeline entrypoint:

```bash
python -m src.data.prepare --config configs/data.yaml
# or
dvc repro prepare
```

### Transformations applied

1. **Ingestion** — Download from Hugging Face with deterministic snapshot ID
2. **Unicode normalization** — NFKC normalization
3. **Control character removal** — Strip non-printable control chars
4. **Zero-width removal** — Remove invisible unicode characters
5. **Whitespace normalization** — Collapse repeated whitespace
6. **Exact deduplication** — SHA-256 fingerprint on normalized text
7. **Near deduplication** — MinHash LSH (threshold configurable, default 0.85)
8. **Label validation** — Validate intent, binary_label, severity, technique
9. **Export** — Parquet files in `data/processed/`
10. **Reporting** — Class distribution + validation + dedup reports

### Configuration

Primary config: `configs/data.yaml`  
DVC params: `params.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Reproducibility seed |
| `lowercase` | false | Preserve case for obfuscation detection |
| `near_duplicate_threshold` | 0.85 | MinHash similarity threshold |
| `drop_invalid_rows` | true | Remove rows failing validation |

## Output Schema

Processed Parquet files retain upstream fields plus:

| Column | Type | Description |
|--------|------|-------------|
| `text_original` | string | Raw prompt before cleaning |
| `text_clean` | string | Cleaned prompt |
| `text` | string | Alias of `text_clean` for training |
| `text_fingerprint` | string | SHA-256 fingerprint for dedup audit |
| `intent_label` | int | 0–6 intent class |
| `binary_label` | int | 0=benign, 1=malicious |
| `severity` | int | threat severity; **1-3 in this snapshot** |
| `technique` | string | Attack technique |
| `ambiguity` | bool | Borderline sample flag |
| `source` | string | Row provenance category (used for subgroup reporting, never as a family key) |

Phase 3.5 outputs additionally carry `row_id` (`<upstream_split>:<index>`), `text_hash`, `upstream_split`,
`family_id`, `split`; train variants carry `parent_row_id` and lineage columns (`augment_method`,
`is_balanced_duplicate`, `is_augmented`, `is_adversarial_paraphrase`).

## Intent Classes

| Label | Intent | Description |
|------:|--------|-------------|
| 0 | benign | Normal user input |
| 1 | direct_injection | Explicit instruction override |
| 2 | system_extraction | System prompt leakage attempt |
| 3 | role_hijack | Persona/role manipulation |
| 4 | obfuscation | Encoded or disguised attack |
| 5 | tool_abuse | Malicious tool/function call |
| 6 | indirect_injection | Context-based injection |

## Validation Rules

- `intent_label` must be in `[0, 6]`
- `intent` string must match `intent_label`
- `binary_label` must be consistent with benign/malicious intent
- `severity` must be in `[1, 3]` (`MIN_SEVERITY`/`MAX_SEVERITY` in `src/data/constants.py`, matching the snapshot)
- Malicious rows must include a non-empty `technique`
- Empty text after cleaning is rejected

## Output Artifacts

```
data/
├── raw/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── snapshot_metadata.json
└── processed/
    ├── train.parquet
    ├── validation.parquet
    ├── test.parquet
    ├── metadata.json
    └── reports/
        ├── class_distribution.json
        ├── class_distribution.md
        ├── validation_report.json
        └── dedup_report.json
```

## Phase 3 Finalization

| Feature | Output |
|---------|--------|
| Technique subsets (encoding, role_play, tool_abuse) | `subsets/technique_*_{split}.parquet` |
| Adversarial paraphrase (intent preserved) | `train_adversarial.parquet` |
| Cross-split leakage audit | `reports/leakage_report.json` |
| Reproducibility appendix | `docs/REPRODUCIBILITY.md` + SHA-256 hashes |
| Kaggle-ready package | `kaggle_package.zip` |

**Audit outcome on the upstream-split artifacts:** `passed: false` - 127 exact (56/63/8) and 125 near (38/46/41)
cross-split pairs. These files are kept for traceability but are **not** valid evaluation data. (Raw-file counts
14/20/4 and processed counts 127/125 are different measurements and must not be added together.)

## Phase 3.5 - Leakage-aware group re-split (`v1_leakfree`)

```bash
python -m src.data.phase35 --config configs/data.yaml      # or: dvc repro resplit
```

Policy: pool all raw rows, build *families* (exact normalised fingerprint ∪ `(example N)` template stem ∪
MinHash-LSH word-3-gram Jaccard ≥ 0.85; rows without shingles never link; `source` is never a key), assign whole
families to one split with stratified greedy placement (~80/10/10, upstream split as tie-break), de-duplicate
(exact + near) inside each split, run a full raw/normalised/near/family audit that **must pass**, then regenerate
`train_balanced` / `train_augmented` / `train_adversarial` from the new train split only and audit each against the
new validation/test.

Measured (`data/processed/splits/v1_leakfree/manifest.json`, built 2026-11-07, seed 42):

| | train | validation | test |
|--|------:|-----------:|-----:|
| after group split | 25,853 | 3,232 | 3,235 |
| exact / near duplicates removed within split | 327 / 3,433 | 51 / 401 | 32 / 401 |
| **final rows** | **22,093** | **2,780** | **2,802** |
| benign / direct / sys-extr / role / obfusc / tool / indirect | 5,431 / 6,317 / 2,084 / 1,784 / 1,844 / 2,432 / 2,201 | 668 / 786 / 254 / 244 / 241 / 307 / 280 | 694 / 813 / 261 / 237 / 237 / 285 / 275 |
| families | 18,826 | 2,386 | 2,385 |
| synthetic rows | 7,388 | 909 | 910 |

Families: 23,597 (22,390 singletons; largest 664; 410 exact, 4,882 template, 3,265 near links; 60 rows without
shingles). 3,508 rows moved relative to their upstream split; 4,645 rows dropped as within-split duplicates
(logged in `dropped_rows.jsonl`). Cross-split overlaps after re-split: raw 0, exact 0, near 0, family 0; the
three train variants also show 0 overlaps with validation/test. Variant sizes: balanced 22,984, augmented 26,116,
adversarial 46,801.

Artifacts: `train|validation|test.parquet`, `train_*.parquet`, `manifest.json` (`frozen: true`, SHA-256 per file,
before/after counts, moved-row matrix, thresholds, licence caveat), `dropped_rows.jsonl`,
`family_assignments.parquet`, `reports/leakage_report*.json`, `subsets/`, `kaggle_package_v1_leakfree.zip`.

Caveat: passing these heuristic checks does not prove zero statistical leakage.

## Phase 2 Enrichment

Phase 2 runs after Phase 1 and produces paper-quality training artifacts.

| Feature | Output |
|---------|--------|
| Obfuscation-aware normalization | `obfuscation_detected`, `obfuscation_types`, `text_obfuscation_aware` |
| Severity bucket analysis | `reports/severity_buckets.json` |
| Ambiguity hard-case subsets | `subsets/ambiguity_{split}.parquet` |
| Minority-class balancing | `train_balanced.parquet` |
| Train-only augmentation | `train_augmented.parquet` |
| MLflow dataset versioning | `mlruns/` experiment `data-engineering` |

Severity buckets (defined for the documented 1-10 scale; on this snapshot only `low_1_2` and `moderate_3_4` are
populated because `severity` ∈ {1, 2, 3}):

- `low_1_2` (1–2)
- `moderate_3_4` (3–4)
- `high_5_6` (5–6) - empty on this snapshot
- `critical_7_10` (7–10) - empty on this snapshot

## Reproducibility

Each run generates:

- **Snapshot ID** — Deterministic hash from dataset identity + split counts + seed
- **metadata.json** — Full pipeline config and per-split statistics
- **DVC tracking** — `dvc repro prepare` reproduces processed artifacts

## Ethical Use & License

- Upstream dataset: **CC BY-NC 4.0** — research/non-commercial use
- Commercial use of upstream data requires permission from Neuralchemy / Sanskar Jajoo
- This repository code: **MIT License**

## Citation

```bibtex
@dataset{jajoo2026threatmatrix,
  author    = {Sanskar Jajoo},
  title     = {Neuralchemy Prompt Injection Threat Matrix},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix}
}
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-08-15 | Phase-1 pipeline: ingest, clean, dedup, validate, parquet export |
| 2.0.0 | 2026-08-15 | Phase-2 enrichment: obfuscation, severity buckets, ambiguity subsets, balancing, augmentation, MLflow |
| 3.0.0 | 2026-08-15 | Phase-3 finalization: technique subsets, adversarial paraphrase, leakage audit, reproducibility, Kaggle package |
| 3.5.0 | 2026-11-07 | Phase-3.5 group-aware leakage-free re-split `v1_leakfree`; measured schema facts (severity 1-3, constant surface, synthetic share); train variants regenerated from the new train split; provenance caveat |
