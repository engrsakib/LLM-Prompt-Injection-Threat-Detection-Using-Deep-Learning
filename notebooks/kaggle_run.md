# Run the study on Kaggle

Use [`kaggle_train.ipynb`](kaggle_train.ipynb) with the source bundle and frozen
Phase 3.5 split files attached as separate inputs. The bootstrap and debug
procedure below is the checklist for verifying the corrected notebook in a fresh Kaggle session.

## 1. Set up the notebook

- Import the updated notebook into a Python session. The study code uses Python 3.11 or newer.
- Select **GPU T4 x2** (or another available CUDA GPU) in **Settings → Accelerator**.
  TextCNN and DistilBERT can also run on CPU for debugging. Use CUDA for full study runs.
  Training uses one GPU per job. The compute-tier check uses each device's total VRAM.
  [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks) ·
  [Kaggle accelerator availability](https://www.kaggle.com/product-announcements/735239)
- Enable **Internet** for package installation, Git cloning if used, and the first download of
  an uncached Hugging Face checkpoint. Keep the runtime's CUDA-enabled PyTorch build.
  For an already provisioned environment, set `INSTALL_DEPS=0` in the environment or
  set `INSTALL_DEPS = False` in the dependency cell.
- Use the environment report in the notebook to check CUDA, available tiers, and free disk.
  Reserve disk space for weights, checkpoints, and artifacts before choosing a model batch.

| Compute tier | Models | Resources |
| --- | --- | --- |
| `cpu_ok` | TextCNN, BiLSTM+Attention, CNN-BiLSTM, DistilBERT, DistilRoBERTa, ALBERT, ELECTRA-small | CPU or CUDA GPU |
| `gpu_small` | BERT, RoBERTa, DeBERTa-v3, SecureBERT, XLNet | At least 6 GB VRAM |
| `gpu_medium` | ModernBERT-large | At least 12 GB VRAM. Use a GPU with bf16 support for the configured precision. |
| `gpu_large_qlora` | Mistral-7B, Qwen3-8B, Llama-3.1-8B | At least 15 GB VRAM and working CUDA-compatible `bitsandbytes` |

Llama-3.1-8B additionally requires approved Hugging Face access. Attach an `HF_TOKEN`
Kaggle Secret, enable its notebook access, and run the following cell before the sweep.
Set `FORCE_BLOCKED = True` in the settings cell for the approved gated job.

```python
import os

from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

## 2. Attach the repository source

From the root of the updated working tree, create a fresh source bundle. Include the README,
which the package metadata references, and the complete `src/` and `configs/` trees.

```bash
zip -r kaggle-repo.zip LICENSE README.md pyproject.toml requirements-kaggle.txt src configs \
  -x '*/__pycache__/*' '*.pyc'
```

On Windows, you can create the bundle from the same directory in PowerShell.

```powershell
Compress-Archive -Path LICENSE,README.md,pyproject.toml,requirements-kaggle.txt,src,configs `
  -DestinationPath kaggle-repo.zip -Force
```

Upload the ZIP as a private Kaggle Dataset and attach it using **Input → Add Input**.
The notebook recognizes an attachment named `kaggle-repo.zip` and extracted source trees.
The repository root inside the bundle contains `pyproject.toml`,
`src/data/phase35.py`, `configs/kaggle_debug.yaml`, `configs/experiment.yaml`, and
`configs/models/`. The split dataset is attached separately. If the notebook sees more
than one source bundle or checkout, set `REPO_SOURCE` in cell 1 to the exact mounted ZIP
or complete repository directory.

For an existing checkout, set `REPO_SOURCE` to its root or start the notebook there.
As another option, set `REPO_URL` in cell 1 to a Git clone URL whose default branch
already has the matching code. Enable Internet for cloning. Record the source revision
and preserve the ZIP if it includes local changes.

## 3. Attach the frozen splits

Package the **existing** frozen Phase 3.5 output from the repository root in your
local project Python environment with the [runtime dependencies](../requirements.txt) installed. Use
`bash scripts/package_kaggle.sh` or its Python entry point. Both accept
`--splits-dir PATH` and `--output PATH` for an alternate location.

```bash
python -m src.data.kaggle_package
python -m zipfile -e data/processed/splits/v1_leakfree/kaggle_upload_v1_leakfree.zip kaggle-splits-upload
```

Upload the extracted files as a separate private Kaggle Dataset and attach the exact
version to the notebook. Keep the following files together in its split directory.

```text
manifest.json
train.parquet
validation.parquet
test.parquet
train_balanced.parquet
train_augmented.parquet
train_adversarial.parquet
```

The packager validates the existing canonical manifest and copies it and its listed
artifacts into a separate upload ZIP. An older `kaggle_package_v1_leakfree.zip` and
its extracted package may contain a provisional manifest with `frozen: false`
even when the six Parquet hashes match.
Repackage from the canonical split directory before uploading. Preserve the original
Parquet bytes and the canonical manifest. Follow the dataset's stated reuse terms.
[Data card](../docs/DATA_CARD.md)

## 4. Bootstrap and validate in order

1. Run the repository setup cell. With no explicit `REPO_SOURCE`, it checks the
   current directory and its parents, attached sources, and direct checkouts under
   `/kaggle/working`, then falls back to a configured Git clone. Check the printed
   repository path.
2. Run the dependency cell. The install uses the active kernel's `sys.executable`
   and constrains torch to the version already installed in the runtime.
3. Run environment and split discovery. The split directory can be read from the Kaggle
   input mount. Set `SPLITS_SOURCE` in the cell to the exact directory if discovery
   finds multiple versions or the files are mounted at an unexpected path.
4. Run the import and frozen-manifest cell. Look for `REPOSITORY IMPORT VERIFIED`,
   which confirms that Phase 3.5 imported from the resolved checkout. The cell uses
   `load_frozen_manifest()` to check the frozen flag, leakage audit, and recorded
   artifact hashes. Proceed when it prints
   `LOCAL FROZEN MANIFEST VALIDATED` for version `v1_leakfree`, with the full-split
   counts of 22,093 train, 2,780 validation, and 2,802 test rows.

## 5. Run the debug profile, then the full study

1. Keep `DEBUG = True`, `MODELS = ["textcnn"]` and `SEEDS = [42]`. Run the
   settings, sweep, tables and figures, and registry cells. Check the sweep shows
   `textcnn/seed_42` as `succeeded`.
2. Set `MODELS = ["distilbert"]` with the same seed. Rerun from the settings cell
   through the registry cell. Check `distilbert/seed_42` also shows `succeeded`.
   The debug config uses 2,000 training rows, 500 rows per evaluation split, two
   TextCNN epochs and one DistilBERT epoch.
3. For the full study, set `DEBUG = False`, `MODELS = None` and `SEEDS = None`.
   This selects `configs/experiment.yaml`, all 16 registered models and seeds
   42, 43, and 44. To fit a session or accelerator tier, set `MODELS` to a batch
   of available model IDs and leave `SEEDS = None` to use all three seeds. Check
   `blocked` jobs and their reasons in the sweep summary.

Set `BUDGET_MIN` to leave time to finish the current job, generate artifacts and
save the session. The budget is checked before a job starts. Within a profile, the
sweep runs the selected jobs, skips successful jobs, and requeues selected jobs
skipped because of the previous budget. It can also resume an interrupted job
from its `last/` checkpoint. If a selected job failed, resolve its error and use
`RETRY_FAILED = True` in the settings cell. Keep the sweep manifest and run directories together.
Debug results are for pipeline verification only.

The Bash runner is also available from a complete checkout with the split input
attached. It runs both debug jobs in order and writes to the debug output tree.
Use the same `SPLITS_SOURCE` and `INSTALL_DEPS` settings when needed. Set
`RETRY_FAILED=1` to retry a failed job after correcting the problem.

```bash
bash scripts/kaggle_run.sh
```

## 6. Verify, save, and restore outputs

Use `/kaggle/working/outputs/debug` for the debug profile and
`/kaggle/working/outputs` for the full profile. Both profiles use the layout below.

```text
runs/<model>/seed_<seed>/
  config.json, history.jsonl, checkpoint/, threshold.json, run_summary.json
  metrics_validation.json, metrics_test.json
  predictions_validation.parquet, predictions_test.parquet
sweep/manifest.json
tables/
  runs_test.csv, runs_test.json, seed_summary_test.csv, seed_summary_test.json
  per_class_test.csv, subgroups_test.csv, bootstrap_ci_test.json, tables_report.json
figures/
  *.png, *.pdf, *.svg, figures_report.json, skipped.json
registry.json
```

Match the run IDs and seeds in the predictions, metrics, and summaries to the
succeeded sweep entries. Confirm that both debug prediction files have 500 rows
per model, the tables contain the evaluated jobs, and the figure report lists
written figures. `figures/skipped.json` gives reasons for unavailable figures, such
as training-variant comparisons with only the base variant present.

Use **Save Version → Save & Run All** to verify a clean top-to-bottom execution in a
separate Kaggle session. Set `MODELS = ["textcnn", "distilbert"]` before saving a
fresh debug version that should contain both jobs. Download the `outputs/` tree from the saved version's
**Output** view. Preserve the profile directories, checkpoints, logs and run settings
with the code and split input versions. [Kaggle notebook versions and outputs](https://www.kaggle.com/docs/notebooks)

To resume from saved artifacts in a later session, attach the prior notebook output
with **Add Input**. Before the sweep cell, copy its output tree into writable working
storage. Replace the example path with the exact mounted `outputs` directory.

```python
from shutil import copytree

copytree(
    "/kaggle/input/<prior-notebook-output>/outputs",
    "/kaggle/working/outputs",
    dirs_exist_ok=True,
)
```

Use matching source, splits, and experiment settings for a resumed job.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Start a fresh kernel and run setup, installation and import verification in order. Confirm the printed module path belongs to the staged repository. |
| `FileNotFoundError` for a repository directory or a different working directory | Attach the code input and let setup resolve the root, or set `REPO_SOURCE` to its exact checkout or ZIP. Paths such as `/kaggle/working/research_project` depend on how the source was staged. |
| Missing `pyproject.toml` or `src/data/phase35.py` | Rebuild the code ZIP from the repository root with the complete source and configuration files. A notebook alone is insufficient. |
| Dependency conflict or import failure after installation | Inspect the pip output and run `subprocess.run([sys.executable, "-m", "pip", "check"])` in the notebook. Restart the kernel after replacing an already-imported dependency. Record the Python and package versions, and keep the Kaggle CUDA torch build. |
| Missing dataset, split directory, or Parquet files | Attach the extracted frozen split dataset and set `SPLITS_SOURCE` to the directory with the manifest and all six Parquets. |
| Manifest not frozen, leakage audit failure, or hash mismatch | Verify the attached split version. Repackage from the existing validated canonical copy or restore that copy from your research archive. Keep the manifest and all Parquets from the same frozen version. |
| DistilBERT checkpoint inaccessible | Enable Internet for the first checkpoint download or make the matching checkpoint available in the Hugging Face cache. |
| Job blocked or output incomplete | Check the sweep manifest and GPU report for the job's status and reason. Confirm training produced a run summary before treating tables or figures as complete. |

## Reproducibility checklist

- Record the repository revision, bundled local changes, notebook version, and Kaggle input versions.
- Preserve the canonical Phase 3.5 manifest and verify all six declared hashes.
- Save the runtime report, Python and package versions, GPU type, and configured compute tier.
- Record the profile, model IDs, seeds, and selected training file.
- Verify saved predictions, metrics, figures, and tables against the sweep and run summaries.
- Keep the profile output tree and any unfinished checkpoints together for resumption.
