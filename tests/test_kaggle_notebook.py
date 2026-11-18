import json
import os
import shutil
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.kaggle.env import find_kaggle_splits_dir

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads(
    (ROOT / "notebooks/kaggle_train.ipynb").read_text(encoding="utf-8")
)
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NOTEBOOK["cells"]}


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "input" / "owner" / "dataset" / "checkout"
    root.mkdir(parents=True)
    for name in ("README.md", "pyproject.toml", "requirements-kaggle.txt"):
        shutil.copy2(ROOT / name, root / name)
    for name in ("src", "configs"):
        shutil.copytree(
            ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    return root


def run_cells(tmp_path, *cell_ids, cwd=None, extra="", **overrides):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "REPO_SOURCE", "REPO_URL", "SPLITS_SOURCE"):
        env.pop(name, None)
    env.update(
        KAGGLE_INPUT_DIR=str(tmp_path / "input"),
        KAGGLE_WORKING_DIR=str(tmp_path / "working"),
    )
    env.update(overrides)
    code = "\n\n".join([*(CELLS[cell] for cell in cell_ids), extra])
    launch_dir = (
        TemporaryDirectory(prefix="kaggle-notebook-cwd-")
        if cwd is None
        else nullcontext(cwd)
    )
    with launch_dir as directory:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )


def test_notebook_bootstrap_resolves_nested_source_and_model_config(tmp_path, checkout):
    result = run_cells(
        tmp_path,
        "repository",
        "settings",
        extra="""
from src.data import phase35
from src.training import train_cli

assert Path(phase35.__file__).resolve() == REPO_DIR / 'src/data/phase35.py'
args = train_cli.build_parser().parse_args(['--model', 'textcnn', '--config', str(CONFIG)])
cfg = train_cli.resolve_config(args)
assert cfg['model']['display_name'] == 'TextCNN'
assert cfg['training']['epochs'] == 2
assert cfg['subset']['train_rows'] == 2000
assert OUTPUT_DIR == WORKING_DIR / 'outputs/debug'
""",
    )
    assert result.returncode == 0, result.stderr
    assert f"Repository ready: {checkout}" in result.stdout


def test_notebook_bootstrap_reuses_direct_checkout_from_nested_cwd(tmp_path, checkout):
    result = run_cells(tmp_path, "repository", cwd=checkout / "configs")
    assert result.returncode == 0, result.stderr
    assert f"Repository ready: {checkout}" in result.stdout
    assert not (tmp_path / "working").exists()


def test_notebook_bootstrap_keeps_relative_source_on_rerun(tmp_path, checkout):
    result = run_cells(
        tmp_path,
        "repository",
        cwd=tmp_path,
        extra=f"exec({CELLS['repository']!r})",
        REPO_SOURCE=str(checkout.relative_to(tmp_path)),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(f"Repository ready: {checkout}") == 2


def test_notebook_bootstrap_stages_nested_zip_and_reuses_it(tmp_path, checkout):
    archive = tmp_path / "input" / "kaggle-repo.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in checkout.rglob("*"):
            if path.is_file():
                bundle.write(
                    path, Path("outer/repository") / path.relative_to(checkout)
                )
    shutil.rmtree(checkout)

    first = run_cells(tmp_path, "repository")
    assert first.returncode == 0, first.stderr
    (staged,) = (tmp_path / "working").iterdir()
    assert (staged / "src/data/phase35.py").is_file()
    second = run_cells(tmp_path, "repository")
    assert second.returncode == 0, second.stderr
    assert second.stdout == first.stdout
    assert list((tmp_path / "working").iterdir()) == [staged]


def test_notebook_bootstrap_reports_partial_and_incorrect_sources(tmp_path, checkout):
    (checkout / "pyproject.toml").unlink()
    result = run_cells(tmp_path, "repository")
    assert result.returncode != 0
    assert "No complete repository found" in result.stderr
    assert "missing pyproject.toml" in result.stderr

    wrong = tmp_path / "working" / "research_project"
    result = run_cells(tmp_path, "repository", REPO_SOURCE=str(wrong))
    assert result.returncode != 0
    assert f"Incomplete repository at {wrong}" in result.stderr
    assert "src/data/phase35.py" in result.stderr
    assert not wrong.exists()


def test_notebook_rejects_path_traversal_in_source_zip(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with zipfile.ZipFile(input_dir / "kaggle-repo.zip", "w") as bundle:
        bundle.writestr("../pyproject.toml", "unexpected")

    result = run_cells(tmp_path, "repository")
    assert result.returncode != 0
    assert "Unsafe path in repository ZIP" in result.stderr
    assert not list((tmp_path / "working").iterdir())


def test_notebook_pip_uses_kernel_and_constrains_torch(tmp_path, checkout):
    result = run_cells(
        tmp_path,
        "repository",
        extra=f"""
from importlib.metadata import version

def verify_pip(command, *, check):
    assert check
    assert command[:5] == [sys.executable, '-m', 'pip', 'install', '-r']
    assert Path(command[5]) == REPO_DIR / 'requirements-kaggle.txt'
    assert command[6] == '-c'
    assert Path(command[7]).read_text().strip() == 'torch==' + version('torch')
    print('Pip invocation verified')

subprocess.run = verify_pip
exec({CELLS["dependencies"]!r})
""",
        REPO_SOURCE=str(checkout),
        INSTALL_DEPS="1",
    )
    assert result.returncode == 0, result.stderr
    assert "Pip invocation verified" in result.stdout


def test_split_override_and_ambiguous_mounts(tmp_path):
    input_dir = tmp_path / "input"
    first = input_dir / "datasets" / "one" / "1"
    second = input_dir / "datasets" / "two" / "2"
    for root in (first, second):
        root.mkdir(parents=True)
        (root / "manifest.json").touch()
        (root / "train.parquet").touch()

    with pytest.raises(ValueError, match="Multiple split bundles.*SPLITS_SOURCE"):
        find_kaggle_splits_dir(input_dir=input_dir)
    assert find_kaggle_splits_dir(second, input_dir=input_dir) == second
    with pytest.raises(FileNotFoundError, match="SPLITS_SOURCE.*manifest.json"):
        find_kaggle_splits_dir(first.parent, input_dir=input_dir)


@pytest.mark.parametrize("prior_state", ["skipped", "failed"])
def test_sweep_resumes_only_selected_jobs(tmp_path, monkeypatch, prior_state):
    from src.training import sweep, train_cli

    manifest_path = tmp_path / "manifest.json"
    manifest = sweep.load_manifest(manifest_path)
    sweep.plan_jobs(manifest, ["distilbert", "textcnn"], [42], sweep.detect())
    manifest["jobs"]["textcnn/seed_42"].update(
        state=prior_state,
        **(
            {"skipped_reason": sweep.BUDGET_SKIP}
            if prior_state == "skipped"
            else {"error": "old error"}
        ),
    )
    sweep.save_manifest(manifest_path, manifest)

    def fake_train(args):
        assert (args.model, args.seed) == ("textcnn", 42)
        return {"run_id": "test-run", "test_f1_macro": 0.5, "wall_time_sec": 0.1}

    monkeypatch.setattr(train_cli, "run", fake_train)
    args = sweep.build_parser().parse_args(
        [
            "--config",
            str(ROOT / "configs/kaggle_debug.yaml"),
            "--manifest",
            str(manifest_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--models",
            "textcnn",
            "--seeds",
            "42",
            *(["--retry-failed"] if prior_state == "failed" else []),
        ]
    )
    jobs = sweep.run_sweep(args)["jobs"]
    assert jobs["textcnn/seed_42"]["state"] == "succeeded"
    assert jobs["distilbert/seed_42"]["state"] == "queued"
