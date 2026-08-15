"""Reproducibility appendix generation with commands and file hashes."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


REPRO_COMMANDS = [
    "pip install -r requirements.txt",
    "python -m src.data.prepare --config configs/data.yaml --phase 1",
    "python -m src.data.enrich --config configs/data.yaml",
    "python -m src.data.finalize --config configs/data.yaml",
    "dvc repro prepare",
    "dvc repro enrich",
    "dvc repro finalize",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifact_hashes(base_dir: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(base_dir.glob(pattern)):
            if path.is_file():
                hashes[str(path.as_posix())] = sha256_file(path)
    return hashes


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_reproducibility_appendix(processed_dir: Path) -> dict:
    """Build reproducibility manifest with environment, commands, and hashes."""
    artifact_hashes = collect_artifact_hashes(
        processed_dir,
        (
            "*.parquet",
            "metadata.json",
            "reports/*.json",
            "subsets/**/*.parquet",
            "kaggle_package/**/*",
        ),
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "commands": REPRO_COMMANDS,
        "artifact_hashes_sha256": artifact_hashes,
    }


def write_reproducibility_appendix(processed_dir: Path, docs_dir: Path) -> dict:
    """Write JSON manifest and Markdown appendix for IEEE reproducibility."""
    manifest = build_reproducibility_appendix(processed_dir)

    json_path = processed_dir / "reports" / "reproducibility_manifest.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_lines = [
        "# Reproducibility Appendix",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{manifest['python_version'].split()[0]}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Git commit: `{manifest.get('git_commit') or 'unavailable'}`",
        "",
        "## Exact Commands",
        "",
    ]
    for cmd in manifest["commands"]:
        md_lines.append(f"```bash\n{cmd}\n```")
        md_lines.append("")

    md_lines.extend(["## Artifact Hashes (SHA-256)", ""])
    for path, digest in sorted(manifest["artifact_hashes_sha256"].items()):
        md_lines.append(f"- `{path}`: `{digest}`")

    md_path = docs_dir / "REPRODUCIBILITY.md"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return manifest
