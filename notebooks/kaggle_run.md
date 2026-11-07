# Running the 16-model study on Kaggle

The Kaggle path is a *thin wrapper*: `notebooks/kaggle_train.ipynb` only calls modules under
`src/`. Everything below has been implemented and unit-tested locally; the Kaggle execution
itself has **not** been performed yet (see `docs/MODEL_REGISTRY.md` for actual statuses).

## 1. What you upload (manual steps, no automation)

| Item | How | Why |
|------|-----|-----|
| Repository | Kaggle Dataset (zip of the repo without `data/`, `outputs/`) **or** `git clone` in cell 1 | code only |
| Split package | Kaggle Dataset from `data/processed/splits/v1_leakfree/kaggle_package_v1_leakfree.zip` | frozen, leakage-audited splits + `manifest.json` |
| `HF_TOKEN` | Kaggle **Secret** (only if you need gated checkpoints, e.g. Llama-3.1-8B) | read via `os.environ`, never written to disk |

Nothing in the code logs in, creates datasets, publishes, submits or calls paid APIs. The dataset
licence (CC BY-NC 4.0 per the local snapshot metadata; upstream page could not be re-verified)
applies to any Kaggle dataset you create - keep it private unless you have verified permission.

## 2. Resource policy

`python -m src.kaggle.env` prints accelerator, VRAM, free disk and the compute tiers that can run:

| Tier | Models | Minimum |
|------|--------|---------|
| `cpu_ok` | TextCNN, BiLSTM+Attn, CNN-BiLSTM, DistilBERT, DistilRoBERTa, ALBERT, ELECTRA-small | CPU (slow) or any GPU |
| `gpu_small` | BERT, RoBERTa, DeBERTa-v3, SecureBERT, XLNet | >= 6 GB VRAM |
| `gpu_medium` | ModernBERT-large | >= 12 GB VRAM, bf16 |
| `gpu_large_qlora` | Mistral-7B, Qwen3-8B, Llama-3.1-8B (4-bit QLoRA) | >= 15 GB VRAM + `bitsandbytes` |

The sweep (`src/training/sweep.py`) marks jobs whose tier is unavailable as `blocked` instead of
crashing; Llama-3.1-8B stays `blocked` until an approved `HF_TOKEN` is present.

## 3. Resumability

* One job = one model x one seed. Trainer checkpoints `last/` every epoch with optimizer,
  scheduler, scaler and RNG state; `--resume` continues from it.
* `outputs/sweep/manifest.json` records `queued | running | succeeded | failed | blocked | skipped`.
  Re-running the notebook skips succeeded jobs and re-queues stale `running` ones.
* `--budget-minutes` stops launching new jobs before the Kaggle session limit (12 h GPU).
* To continue across sessions: add the previous notebook's **output** as an input dataset and copy
  `outputs/` back to `/kaggle/working/outputs` before cell 6 (Kaggle inputs are read-only).

## 4. Debug profile first

`configs/kaggle_debug.yaml` (TextCNN + DistilBERT, 2,000 train rows, 1-2 epochs) verifies the full
path - training, validation-selected threshold, per-example predictions, metrics, tables, figures -
in minutes. Switch `DEBUG = False` in cell 5 for `configs/experiment.yaml` (3 seeds, full splits).

## 5. Artifacts produced (all under `/kaggle/working/outputs`)

```
runs/<model>/seed_<s>/{config.json,history.jsonl,checkpoint/,metrics_validation.json,
                      metrics_test.json,predictions_validation.parquet,predictions_test.parquet,
                      threshold.json,run_summary.json}
sweep/manifest.json
tables/{runs_test.csv,seed_summary_test.csv,per_class_test.csv,subgroups_test.csv,bootstrap_ci_test.json}
figures/*.png|pdf|svg + skipped.json
registry.json
```

Download them from the Output tab and copy into the local repo's `outputs/` to regenerate
documentation tables (`python -m src.models.registry`, `python -m src.eval.tables`, `python -m src.eval.plots`).

## 6. Local equivalents (verified on this machine, CPU only)

```bash
python -m src.data.phase35 --config configs/data.yaml                 # Phase 3.5 re-split (90 s)
python -m src.training.train_cli --model textcnn    --seed 42 --smoke # ~10 s
python -m src.training.train_cli --model distilbert --seed 42 --smoke # minutes on CPU (downloads 268 MB once)
python -m src.eval.tables --runs outputs/smoke --out outputs/smoke/tables
python -m src.eval.plots  --runs outputs/smoke --out outputs/smoke/figures --tag "SMOKE"
```
