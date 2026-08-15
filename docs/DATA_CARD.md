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

The upstream Threat Matrix dataset contains **32,320** curated prompt-injection samples with:

- 7 intent classes
- Binary malicious label
- Severity score (1–10)
- Technique and surface metadata
- Ambiguity flag for borderline cases

Official splits: **80% train / 10% validation / 10% test** (maintained during processing).

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
| `severity` | int | 1–10 threat severity |
| `technique` | string | Attack technique |
| `ambiguity` | bool | Borderline sample flag |

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
- `severity` must be in `[1, 10]`
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
