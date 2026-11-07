"""Resumable model x seed sweep driven by a JSON manifest.

Each job = one model x one seed, executed in-process via ``src.training.train_cli.run``.
The manifest (``outputs/sweep/manifest.json``) tracks every job's state:

    queued | running | succeeded | failed | blocked | skipped

Re-running the same command resumes: succeeded jobs are skipped, ``running`` jobs left by a
killed session are re-queued (and the trainer itself resumes from ``last/``), ``blocked`` jobs
(gated checkpoints, insufficient accelerator tier) stay blocked with their reason. A wall-clock
budget stops the sweep cleanly before Kaggle's session limit.

CLI::

    python -m src.training.sweep --config configs/experiment.yaml --models textcnn distilbert --seeds 42 43
    python -m src.training.sweep --config configs/kaggle_debug.yaml --budget-minutes 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.kaggle.env import TIER_MIN_VRAM_GB, detect
from src.models.registry import get_spec, list_model_ids

STATES = ("queued", "running", "succeeded", "failed", "blocked", "skipped")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": _now(), "jobs": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def job_key(model_id: str, seed: int) -> str:
    return f"{model_id}/seed_{seed}"


def plan_jobs(manifest: dict[str, Any], models: list[str], seeds: list[int], env, force_blocked: bool = False) -> None:
    """Add missing jobs; mark blocked ones; re-queue stale running jobs."""
    for model_id in models:
        spec = get_spec(model_id)
        for seed in seeds:
            key = job_key(model_id, seed)
            job = manifest["jobs"].get(key, {"model_id": model_id, "seed": seed, "state": "queued", "attempts": 0, "history": []})
            if job["state"] == "running":
                job["state"] = "queued"
                job["history"].append({"at": _now(), "event": "stale running job re-queued"})
            if job["state"] in ("queued", "blocked"):
                reason = None
                if spec.status == "blocked" and not force_blocked:
                    reason = f"registry: {spec.access_note}"
                elif spec.compute_tier not in env.recommended_tiers:
                    reason = f"compute tier '{spec.compute_tier}' not available (have: {env.recommended_tiers}; need >= {TIER_MIN_VRAM_GB.get(spec.compute_tier)} GB VRAM)"
                elif spec.gated and not env.hf_token_present:
                    reason = "gated checkpoint and no HF_TOKEN in environment"
                if reason:
                    job["state"], job["blocked_reason"] = "blocked", reason
                else:
                    job["state"] = "queued"
                    job.pop("blocked_reason", None)
            manifest["jobs"][key] = job


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    models = args.models or cfg.get("models") or list_model_ids()
    seeds = args.seeds or cfg.get("seeds") or [42, 43, 44]
    outputs = cfg.get("outputs", {})
    runs_dir = Path(args.runs_dir or outputs.get("runs_dir", "outputs/runs"))
    manifest_path = Path(args.manifest or outputs.get("sweep_manifest", "outputs/sweep/manifest.json"))
    env = detect(runs_dir.parent)
    manifest = load_manifest(manifest_path)
    manifest["environment"] = env.to_dict()
    manifest["config"] = str(args.config)
    plan_jobs(manifest, models, [int(s) for s in seeds], env, force_blocked=args.force_blocked)
    save_manifest(manifest_path, manifest)

    from src.training import train_cli

    deadline = time.time() + args.budget_minutes * 60 if args.budget_minutes else None
    for key, job in manifest["jobs"].items():
        if job["state"] != "queued":
            continue
        if deadline and time.time() > deadline:
            job["state"] = "skipped"
            job["skipped_reason"] = "wall-clock budget exhausted; re-run to continue"
            continue
        if args.dry_run:
            continue
        job.update({"state": "running", "started_at": _now()})
        job["attempts"] += 1
        save_manifest(manifest_path, manifest)
        cli_args = [
            "--model", job["model_id"], "--seed", str(job["seed"]), "--config", str(args.config),
            "--output-dir", str(runs_dir), "--resume",
        ]
        if args.splits_dir:
            cli_args += ["--splits-dir", args.splits_dir]
        if args.train_file:
            cli_args += ["--train-file", args.train_file]
        if args.offline:
            cli_args.append("--offline")
        if args.force_blocked:
            cli_args.append("--force-blocked")
        if args.smoke:
            cli_args.append("--smoke")
        try:
            summary = train_cli.run(train_cli.build_parser().parse_args(cli_args))
            job.update({"state": "succeeded", "finished_at": _now(), "run_id": summary["run_id"], "test_f1_macro": summary["test_f1_macro"], "wall_time_sec": summary["wall_time_sec"]})
        except SystemExit as exc:  # blocked model refused by train_cli
            job.update({"state": "blocked", "blocked_reason": str(exc), "finished_at": _now()})
        except Exception as exc:  # noqa: BLE001
            job.update({"state": "failed", "finished_at": _now(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
            if "out of memory" in str(exc).lower():
                job["blocked_reason"] = "CUDA OOM - reduce batch_size / max_length or move to a larger tier"
        save_manifest(manifest_path, manifest)

    counts = {s: sum(1 for j in manifest["jobs"].values() if j["state"] == s) for s in STATES}
    manifest["summary"] = counts
    save_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable model x seed sweep.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--models", nargs="*", default=None, choices=list_model_ids())
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--train-file", default=None)
    p.add_argument("--budget-minutes", type=float, default=None, help="Stop launching new jobs after this many minutes")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--force-blocked", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Only plan / update the manifest")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_sweep(args)
    print(json.dumps(manifest["summary"], indent=2))
    for key, job in manifest["jobs"].items():
        extra = job.get("blocked_reason") or job.get("error") or job.get("skipped_reason") or ""
        print(f"{job['state']:<10} {key:<28} {extra[:100]}")
    return 0 if manifest["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
