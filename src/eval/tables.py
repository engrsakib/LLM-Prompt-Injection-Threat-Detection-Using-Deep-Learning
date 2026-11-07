"""Aggregate per-run metrics into study-level tables.

Two kinds of uncertainty are reported SEPARATELY and never mixed:

* **seed summary** – mean / std / min / max across seeds of the same model (variance from
  initialisation + data order);
* **group-aware bootstrap CI** – percentile CI from resampling *families* (``family_id``) of the
  test set for a single run's predictions (variance from the finite test sample, respecting the
  duplicate/template structure).

Inputs are the artifacts written by ``src.eval.predict`` under ``outputs/runs`` (or a smoke dir).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import LABEL_ORDER
from src.eval.evaluate import binary_metrics, multiclass_metrics
from src.eval.predict import PROB_COLUMNS, load_predictions

SCALAR_KEYS_MC = ("accuracy", "balanced_accuracy", "f1_macro", "f1_micro", "f1_weighted", "precision_macro", "recall_macro", "mcc", "auroc_ovr_macro", "ap_ovr_macro")
SCALAR_KEYS_BIN = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate", "mcc", "auroc", "average_precision")


def discover_runs(runs_dir: str | Path, split: str = "test") -> list[Path]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(p.parent for p in runs_dir.rglob(f"metrics_{split}.json"))


def collect_run_metrics(runs_dir: str | Path, split: str = "test") -> pd.DataFrame:
    rows = []
    for run_dir in discover_runs(runs_dir, split):
        payload = json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8"))
        summary_path = run_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        mc, bi = payload["multiclass"], payload["binary"]
        row: dict[str, Any] = {
            "model_id": payload.get("model_id"),
            "seed": payload.get("seed"),
            "run_id": payload.get("run_id"),
            "run_dir": str(run_dir.as_posix()),
            "split": split,
            "n": mc["n"],
            "smoke": summary.get("smoke"),
            "wall_time_sec": summary.get("train_wall_time_sec"),
            "params_total": (summary.get("params") or {}).get("params_total"),
            "params_trainable": (summary.get("params") or {}).get("params_trainable"),
            "binary_threshold": bi["threshold"],
        }
        for key in SCALAR_KEYS_MC:
            row[f"mc_{key}"] = mc.get(key)
        for key in SCALAR_KEYS_BIN:
            row[f"bin_{key}"] = bi.get(key)
        for name in LABEL_ORDER:
            row[f"f1_{name}"] = mc["per_class"][name]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(run_table: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max across seeds per model. Requires >=1 run; std is NaN with a single seed."""
    if run_table.empty:
        return pd.DataFrame()
    metric_cols = [c for c in run_table.columns if c.startswith(("mc_", "bin_", "f1_"))]
    agg = run_table.groupby("model_id")[metric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = run_table.groupby("model_id")["seed"].nunique()
    agg["seeds"] = run_table.groupby("model_id")["seed"].agg(lambda s: ",".join(str(int(x)) for x in sorted(s)))
    return agg.reset_index()


def group_bootstrap_ci(
    preds: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    group_col: str = "family_id",
    metrics: tuple[str, ...] = ("f1_macro", "accuracy", "balanced_accuracy"),
    binary_metrics_keys: tuple[str, ...] = ("f1", "auroc"),
) -> dict[str, Any]:
    """Percentile bootstrap over groups (families). Falls back to row bootstrap when no group column."""
    rng = np.random.default_rng(seed)
    y = preds["gold_label"].to_numpy(dtype=int)
    probs = preds[PROB_COLUMNS].to_numpy(dtype=float)
    threshold = float(preds["binary_threshold"].iloc[0])
    if group_col in preds.columns and preds[group_col].notna().any():
        fallback = pd.Series([f"__row__{i}" for i in range(len(preds))], index=preds.index)
        groups = preds[group_col].astype(object).where(preds[group_col].notna(), fallback).astype(str).to_numpy()
        unit = "family"
    else:
        groups = np.arange(len(preds)).astype(str)
        unit = "row"
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    samples: dict[str, list[float]] = {f"mc_{m}": [] for m in metrics}
    samples.update({f"bin_{m}": [] for m in binary_metrics_keys})
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[c] for c in chosen])
        try:
            mc = multiclass_metrics(y[idx], probs[idx])
            bi = binary_metrics(y[idx], probs[idx], threshold)
        except Exception:  # noqa: BLE001 - degenerate resample (e.g. single class)
            continue
        for m in metrics:
            if mc.get(m) is not None:
                samples[f"mc_{m}"].append(mc[m])
        for m in binary_metrics_keys:
            if bi.get(m) is not None:
                samples[f"bin_{m}"].append(bi[m])
    out: dict[str, Any] = {"unit": unit, "n_groups": int(len(uniq)), "n_rows": int(len(preds)), "n_boot": n_boot, "alpha": alpha, "ci": {}}
    for key, values in samples.items():
        if len(values) < 10:
            out["ci"][key] = {"low": None, "high": None, "n_valid": len(values), "reason": "too few valid resamples"}
            continue
        arr = np.asarray(values)
        out["ci"][key] = {"low": float(np.quantile(arr, alpha / 2)), "high": float(np.quantile(arr, 1 - alpha / 2)), "mean": float(arr.mean()), "n_valid": len(values)}
    return out


def per_class_table(runs_dir: str | Path, split: str = "test") -> pd.DataFrame:
    rows = []
    for run_dir in discover_runs(runs_dir, split):
        payload = json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8"))
        mc = payload["multiclass"]
        for name in LABEL_ORDER:
            pc, counts = mc["per_class"][name], mc["per_class_counts"][name]
            rows.append({"model_id": payload["model_id"], "seed": payload["seed"], "class": name, **pc, **{k: counts[k] for k in ("tp", "fp", "fn", "tn")}, "auroc_ovr": mc["auroc_ovr_per_class"][name], "ap_ovr": mc["ap_ovr_per_class"][name]})
    return pd.DataFrame(rows)


def subgroup_table(runs_dir: str | Path, split: str = "test", by: tuple[str, ...] = ("source", "technique", "ambiguity_bucket", "severity")) -> pd.DataFrame:
    """Macro-F1 / accuracy per metadata subgroup from saved predictions (synthetic vs non-synthetic included)."""
    rows = []
    for run_dir in discover_runs(runs_dir, split):
        try:
            preds = load_predictions(run_dir, split)
        except FileNotFoundError:
            continue
        preds = preds.copy()
        if "source" in preds.columns:
            preds["synthetic"] = preds["source"].astype(str).str.startswith("synthetic").map({True: "synthetic", False: "non_synthetic"})
        for col in (*by, "synthetic"):
            if col not in preds.columns or preds[col].isna().all():
                continue
            for value, part in preds.groupby(col):
                y = part["gold_label"].to_numpy(dtype=int)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # subsets legitimately miss classes
                        mc = multiclass_metrics(y, part[PROB_COLUMNS].to_numpy())
                except Exception:  # noqa: BLE001
                    continue
                rows.append({"model_id": preds["model_id"].iloc[0], "seed": int(preds["seed"].iloc[0]), "group_by": col, "group": str(value), "n": len(part), "f1_macro": mc["f1_macro"], "accuracy": mc["accuracy"], "classes_missing": len(mc["classes_missing_in_gold"])})
    return pd.DataFrame(rows)


def build_tables(runs_dir: str | Path, out_dir: str | Path, split: str = "test", n_boot: int = 1000) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_table = collect_run_metrics(runs_dir, split)
    report: dict[str, Any] = {"runs_dir": str(Path(runs_dir).as_posix()), "split": split, "n_runs": int(len(run_table)), "files": []}
    if run_table.empty:
        (out_dir / "README.txt").write_text("No evaluated runs found. Train models first (src.training.train_cli).", encoding="utf-8")
        report["note"] = "no runs"
        return report
    run_table.to_csv(out_dir / f"runs_{split}.csv", index=False)
    run_table.to_json(out_dir / f"runs_{split}.json", orient="records", indent=2)
    summary = seed_summary(run_table)
    summary.to_csv(out_dir / f"seed_summary_{split}.csv", index=False)
    summary.to_json(out_dir / f"seed_summary_{split}.json", orient="records", indent=2)
    per_class_table(runs_dir, split).to_csv(out_dir / f"per_class_{split}.csv", index=False)
    subgroup_table(runs_dir, split).to_csv(out_dir / f"subgroups_{split}.csv", index=False)
    cis = {}
    for run_dir in discover_runs(runs_dir, split):
        try:
            preds = load_predictions(run_dir, split)
        except FileNotFoundError:
            continue
        key = f"{preds['model_id'].iloc[0]}/seed_{int(preds['seed'].iloc[0])}"
        cis[key] = group_bootstrap_ci(preds, n_boot=n_boot)
    (out_dir / f"bootstrap_ci_{split}.json").write_text(json.dumps(cis, indent=2), encoding="utf-8")
    report["files"] = sorted(p.name for p in out_dir.iterdir())
    report["models"] = sorted(run_table["model_id"].unique().tolist())
    (out_dir / "tables_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate run metrics into tables (CSV + JSON).")
    p.add_argument("--runs", default="outputs/runs")
    p.add_argument("--out", default="outputs/tables")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv)
    report = build_tables(args.runs, args.out, args.split, args.n_boot)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
