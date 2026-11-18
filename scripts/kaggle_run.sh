#!/usr/bin/env bash
# Run both debug jobs from a complete checkout with an attached frozen split bundle.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo_dir"
python_bin="${PYTHON:-python}"
input_dir="${KAGGLE_INPUT_DIR:-/kaggle/input}"
working_dir="${KAGGLE_WORKING_DIR:-/kaggle/working}"
output_dir="$working_dir/outputs/debug"
runs_dir="$output_dir/runs"

if [[ "${INSTALL_DEPS:-1}" == 1 ]]; then
  "$python_bin" - <<'PY'
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as temp:
    constraint = Path(temp) / "torch.txt"
    constraint.write_text(f"torch=={version('torch')}\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements-kaggle.txt", "-c", str(constraint)],
        check=True,
    )
PY
fi

"$python_bin" -m src.kaggle.env --working-dir "$working_dir"
splits_dir="$("$python_bin" - "${SPLITS_SOURCE:-}" "$input_dir" <<'PY'
import sys
from pathlib import Path

from src.kaggle.env import find_kaggle_splits_dir, validate_kaggle_splits

splits = find_kaggle_splits_dir(sys.argv[1], input_dir=Path(sys.argv[2]))
validate_kaggle_splits(splits)
print("LOCAL FROZEN MANIFEST VALIDATED", file=sys.stderr)
print(splits)
PY
)"

sweep_args=(
  --config configs/kaggle_debug.yaml --splits-dir "$splits_dir"
  --runs-dir "$runs_dir" --manifest "$output_dir/sweep/manifest.json" --budget-minutes "${BUDGET_MIN:-480}"
)
if [[ "${RETRY_FAILED:-0}" == 1 ]]; then
  sweep_args+=(--retry-failed)
fi

"$python_bin" -m src.training.sweep "${sweep_args[@]}"
"$python_bin" -m src.eval.tables --runs "$runs_dir" --out "$output_dir/tables" --n-boot 500
"$python_bin" -m src.eval.plots --runs "$runs_dir" --out "$output_dir/figures" --splits-dir "$splits_dir" --tag DEBUG
"$python_bin" -m src.models.registry --runs-dir "$runs_dir" --smoke-dir "$output_dir/smoke" \
  --export "$output_dir/registry.json" --markdown ""
