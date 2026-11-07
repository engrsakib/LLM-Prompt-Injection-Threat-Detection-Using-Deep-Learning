# Reproducibility Appendix

Last updated: 2026-11-07 (Phase 3.5 + 16-model study scaffolding). Only commands that were actually executed on
the development machine are listed as *verified*; everything else is marked *not yet executed*.

> Note: `python -m src.data.finalize` (Phase 3) regenerates this file from `src/data/reproducibility.py` with the
> Phase 1-3 content only. Re-apply the Phase 3.5 / study sections from git history if you re-run finalize.

## Environment (development machine)

- Python `3.13.7`, Windows 11 (`10.0.22631`), **no CUDA** (CPU-only torch `2.13.0+cpu`)
- transformers `5.22.2`, peft `0.21.2`, datasets `5.1.0`, huggingface_hub `2.4.0`, datasketch, pandas, pyarrow,
  scikit-learn, matplotlib
- Pinned requirement files: `requirements.txt` (runtime), `requirements-dev.txt` (tests/lint),
  `requirements-kaggle.txt` (Kaggle session pins; no torch re-pin)

## Verified commands (executed 2026-11-07)

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m compileall -q src tests                              # OK
pytest -q                                                      # 42 passed
python -m src.data.phase35 --config configs/data.yaml          # ~90 s; leakage passed: True; 22,093 / 2,780 / 2,802 rows
python -m src.training.train_cli --model textcnn    --seed 42 --smoke   # ~10 s, outputs/smoke/textcnn_seed42
python -m src.training.train_cli --model distilbert --seed 42 --smoke   # ~5 min CPU, outputs/smoke/distilbert_seed42
python -m src.training.sweep --config configs/experiment.yaml --models textcnn --seeds 7 --smoke \
       --runs-dir outputs/smoke --manifest outputs/smoke/sweep_manifest.json   # run twice: second run skips the succeeded job
python -m src.eval.tables --runs outputs/smoke --out outputs/smoke/tables --n-boot 200
python -m src.eval.plots  --runs outputs/smoke --out outputs/smoke/figures --tag "SMOKE - not for publication"  # 27 written, 1 skipped (ablation)
python -m src.models.registry --check                          # 16 entries, transformers version OK for all
python -m src.kaggle.env                                       # cpu_ok tier only on this machine
```

Smoke runs use 300 stratified training rows, 100 validation/test rows, 1 epoch, max_length 64. They verify the
pipeline only and are **not** results.

## Not yet executed

```bash
# Production training (GPU required; one job per model x seed) - see notebooks/kaggle_run.md
python -m src.training.sweep --config configs/experiment.yaml                 # all 16 models x seeds 42 43 44
python -m src.training.train_cli --model <model_id> --seed <seed> --config configs/experiment.yaml
python -m src.eval.tables --runs outputs/runs --out outputs/tables
python -m src.eval.plots  --runs outputs/runs --out outputs/figures
python -m src.models.registry --check --online                                # Hub metadata verification (network)
dvc repro resplit                                                             # DVC stage for Phase 3.5 (same command as above)
```

Hub metadata (sha / gating / parameter counts) for the registered checkpoints was verified once via the
Hugging Face API during the audit on 2026-11-07 and pinned in `src/models/registry.py`.

## Data pipeline commands (Phases 1-3, executed 2026-08-15)

```bash
python -m src.data.prepare --config configs/data.yaml --phase 1
python -m src.data.enrich --config configs/data.yaml
python -m src.data.finalize --config configs/data.yaml
dvc repro prepare && dvc repro enrich && dvc repro finalize
```

Their artifacts under `data/processed/` are historical; the Phase 3 leakage audit on them did **not** pass and
they are not used for evaluation.

## Determinism notes

- Phase 3.5 is deterministic given the raw files, `configs/data.yaml -> phase35` and `seed` (greedy group
  placement, MinHash with fixed permutations); artifact SHA-256 hashes are stored in
  `data/processed/splits/v1_leakfree/manifest.json` and re-checked by every training/evaluation entrypoint.
- Training seeds control Python/NumPy/torch RNGs and the DataLoader generator; CPU runs are bit-reproducible,
  CUDA runs are reproducible up to kernel nondeterminism (documented in `run_summary.json -> hardware`).
- Binary thresholds are always selected on validation and stored in `threshold.json`.

## Artifact hashes (SHA-256) - Phase 3.5 `v1_leakfree`

See `data/processed/splits/v1_leakfree/manifest.json -> artifact_hashes_sha256` (the directory is gitignored; the
manifest is the source of truth and is included in the Kaggle package). Historical Phase 1-3 hashes are preserved
in `data/processed/reports/reproducibility_manifest.json`.
