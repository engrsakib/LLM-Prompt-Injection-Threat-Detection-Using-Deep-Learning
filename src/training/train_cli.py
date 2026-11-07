"""Train + evaluate ONE registered model for ONE seed.

Examples
--------
Smoke test on CPU (tiny stratified subset, writes to ``outputs/smoke/``)::

    python -m src.training.train_cli --model textcnn --seed 42 --smoke
    python -m src.training.train_cli --model distilbert --seed 42 --smoke

Full run (GPU recommended; writes to ``outputs/runs/<model>/seed_<seed>/``)::

    python -m src.training.train_cli --model roberta --seed 42 --config configs/experiment.yaml

Resume an interrupted run::

    python -m src.training.train_cli --model roberta --seed 42 --resume

Validation is used only for early stopping and binary-threshold selection. Test predictions
are produced once, from the best validation checkpoint. The data manifest must be frozen and
must have passed the leakage audit (``src.data.phase35``), otherwise the run refuses to start.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.data.constants import LABEL_ORDER
from src.models.registry import build_model, get_spec, list_model_ids, load_model
from src.training.data import LABEL_COL, TEXT_COL, load_splits, stratified_subset
from src.training.trainer import TrainConfig, Trainer, hardware_info, pick_device, predict_probabilities, set_seed

DEFAULT_EXPERIMENT = Path("configs") / "experiment.yaml"
MODEL_CONFIG_DIR = Path("configs") / "models"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """experiment.yaml <- configs/models/<model>.yaml <- CLI overrides."""
    cfg = _read_yaml(Path(args.config))
    spec = get_spec(args.model)
    model_cfg = _read_yaml(MODEL_CONFIG_DIR / f"{args.model}.yaml")
    train_hp = dict(spec.hparams)
    train_hp.update(model_cfg.get("training", {}))
    train_hp.update(cfg.get("training_overrides", {}).get(args.model, {}))
    for key in ("epochs", "batch_size", "learning_rate", "max_length", "grad_accum", "precision"):
        value = getattr(args, key, None)
        if value is not None:
            train_hp[key] = value
    cfg["training"] = train_hp
    cfg["model"] = {**model_cfg.get("model", {}), "model_id": args.model}
    cfg.setdefault("data", {})
    if args.splits_dir:
        cfg["data"]["splits_dir"] = args.splits_dir
    if args.train_file:
        cfg["data"]["train_file"] = args.train_file
    cfg["data"].setdefault("splits_dir", "data/processed/splits/v1_leakfree")
    cfg["data"].setdefault("train_file", "train.parquet")
    cfg["seed"] = args.seed
    if args.smoke:
        smoke = cfg.get("smoke", {})
        cfg["training"].update(
            {
                "epochs": int(smoke.get("epochs", 1)),
                "max_length": int(smoke.get("max_length", 64)),
                "batch_size": int(smoke.get("batch_size", 16)),
                "early_stopping_patience": 0,
                "log_every": 5,
            }
        )
        cfg["smoke"] = {
            "train_rows": int(smoke.get("train_rows", 300)),
            "eval_rows": int(smoke.get("eval_rows", 100)),
            **{k: v for k, v in smoke.items() if k not in ("train_rows", "eval_rows")},
        }
    return cfg


def train_config_from(cfg: dict[str, Any], seed: int) -> TrainConfig:
    hp = cfg["training"]
    allowed = TrainConfig.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in hp.items() if k in allowed}
    kwargs["seed"] = seed
    return TrainConfig(**kwargs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_config(args)
    spec = get_spec(args.model)
    if spec.status == "blocked" and not args.force_blocked:
        raise SystemExit(
            f"Model '{spec.model_id}' is registered as BLOCKED: {spec.access_note}\n"
            "Provide access (e.g. HF_TOKEN env var with approved access) and pass --force-blocked."
        )
    seed = int(args.seed)
    set_seed(seed)

    base_out = Path(args.output_dir) if args.output_dir else (Path("outputs") / ("smoke" if args.smoke else "runs"))
    run_dir = base_out / (f"{spec.model_id}_seed{seed}" if args.smoke else f"{spec.model_id}/seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{spec.model_id}-s{seed}-{uuid.uuid4().hex[:8]}"
    t_start = time.time()

    bundle = load_splits(cfg["data"]["splits_dir"], cfg["data"]["train_file"], allow_unfrozen=bool(args.allow_unfrozen))
    train_df, val_df, test_df = bundle.train, bundle.validation, bundle.test
    subset = cfg.get("smoke") if args.smoke else cfg.get("subset")  # kaggle_debug.yaml uses `subset`
    if subset and subset.get("train_rows"):
        train_df = stratified_subset(train_df, int(subset["train_rows"]), seed)
        val_df = stratified_subset(val_df, int(subset.get("eval_rows", len(val_df))), seed)
        test_df = stratified_subset(test_df, int(subset.get("eval_rows", len(test_df))), seed)

    train_cfg = train_config_from(cfg, seed)
    device = pick_device()
    cache_dir = os.environ.get("HF_HOME") or cfg.get("hf_cache_dir")
    local_only = bool(args.offline or os.environ.get("HF_HUB_OFFLINE") == "1")

    model = build_model(
        spec,
        train_texts=train_df[TEXT_COL].tolist(),
        hparams=cfg["training"],
        cache_dir=cache_dir,
        local_files_only=local_only,
        checkpoint_override=cfg["model"].get("checkpoint_override"),
    )
    params = model.count_parameters()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": spec.to_dict() | params,
                "training": train_cfg.to_dict(),
                "data": {
                    "splits_dir": str(bundle.splits_dir.as_posix()),
                    "train_file": bundle.train_file,
                    "manifest_version": bundle.manifest.get("version"),
                    "manifest_leakage_passed": bundle.manifest.get("leakage_passed"),
                    "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
                    "smoke": bool(args.smoke),
                    "subset": subset or None,
                },
                "label_order": list(LABEL_ORDER),
                "hardware": hardware_info(),
                "argv": sys.argv,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    trainer = Trainer(model, train_cfg, run_dir, device=device)
    result = trainer.fit(
        train_df[TEXT_COL].tolist(),
        train_df[LABEL_COL].tolist(),
        val_df[TEXT_COL].tolist(),
        val_df[LABEL_COL].tolist(),
        resume=bool(args.resume),
    )

    # Reload the best validation checkpoint before scoring anything.
    best_model = load_model(spec, trainer.checkpoint_dir) if (trainer.checkpoint_dir / "classifier_meta.json").exists() else model
    val_probs, val_trunc = predict_probabilities(best_model, val_df[TEXT_COL].tolist(), train_cfg.eval_batch_size, train_cfg.max_length, device, train_cfg.precision)
    test_probs, test_trunc = predict_probabilities(best_model, test_df[TEXT_COL].tolist(), train_cfg.eval_batch_size, train_cfg.max_length, device, train_cfg.precision)

    from src.eval.predict import evaluate_run

    eval_results = evaluate_run(
        run_dir,
        spec.model_id,
        seed,
        run_id,
        val_df,
        val_probs,
        test_df,
        test_probs,
        extra={"truncation_rate": {"train": result.train_truncation_rate, "validation": val_trunc, "test": test_trunc}, "max_length": train_cfg.max_length},
    )

    summary = {
        "run_id": run_id,
        "model_id": spec.model_id,
        "seed": seed,
        "smoke": bool(args.smoke),
        "subset": subset or None,
        "status": "succeeded",
        "wall_time_sec": time.time() - t_start,
        "train_wall_time_sec": result.wall_time_sec,
        "epochs_completed": result.epochs_completed,
        "global_step": result.global_step,
        "stopped_early": result.stopped_early,
        "best_val_f1_macro": result.best_metric,
        "best_epoch": result.best_epoch,
        "truncation_rate": {"train": result.train_truncation_rate, "validation": val_trunc, "test": test_trunc},
        "max_length": train_cfg.max_length,
        "params": params,
        "hardware": result.hardware,
        "platform_node": platform.node(),
        "test_f1_macro": eval_results["test"]["multiclass"]["f1_macro"],
        "test_accuracy": eval_results["test"]["multiclass"]["accuracy"],
        "test_binary_f1": eval_results["test"]["binary"]["f1"],
        "binary_threshold": eval_results["threshold"]["threshold"],
        "data": {"splits_dir": str(bundle.splits_dir.as_posix()), "train_file": bundle.train_file, "manifest_version": bundle.manifest.get("version")},
        "artifacts": sorted(p.name for p in run_dir.iterdir()),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not args.keep_last:
        # 'last' duplicates the final-epoch weights; keep disk usage down once the run succeeded.
        import shutil

        shutil.rmtree(trainer.last_dir, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ("run_id", "model_id", "seed", "epochs_completed", "best_val_f1_macro", "test_f1_macro", "test_accuracy", "wall_time_sec")}, indent=2))
    print(f"Artifacts -> {run_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_EXPERIMENT))
    p.add_argument("--model", required=True, choices=list_model_ids())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--train-file", default=None, help="train.parquet | train_balanced.parquet | train_augmented.parquet | train_adversarial.parquet")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny CPU-friendly run written to outputs/smoke/")
    p.add_argument("--offline", action="store_true", help="HF local_files_only")
    p.add_argument("--allow-unfrozen", action="store_true", help="Skip the frozen-manifest/leakage gate (debug only)")
    p.add_argument("--force-blocked", action="store_true")
    p.add_argument("--keep-last", action="store_true")
    for key, typ in (("epochs", int), ("batch_size", int), ("learning_rate", float), ("max_length", int), ("grad_accum", int), ("precision", str)):
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, type=typ, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
