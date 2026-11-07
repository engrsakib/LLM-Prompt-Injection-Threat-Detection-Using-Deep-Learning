"""Per-example prediction export and per-run evaluation.

Each run directory (``outputs/runs/<model_id>/seed_<seed>/`` or ``outputs/smoke/...``) gets:

* ``predictions_validation.parquet`` / ``predictions_test.parquet`` – one row per example with
  id/hash, split, gold label, predicted label, all 7 probabilities, binary prob / pred, metadata
  (source, technique, ambiguity, severity, family_id), model id, seed, run id;
* ``metrics_validation.json`` / ``metrics_test.json`` – the full metric bundle incl. the
  validation-selected binary threshold;
* ``threshold.json`` – threshold selection record (validation only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import LABEL_ORDER
from src.eval.evaluate import attack_probability, evaluate_split, select_binary_threshold, validate_probabilities

PROB_COLUMNS = [f"prob_{name}" for name in LABEL_ORDER]
META_COLUMNS = ("source", "technique", "ambiguity_bucket", "severity", "family_id", "parent_row_id", "upstream_split")


def predictions_frame(
    frame: pd.DataFrame,
    probs: np.ndarray,
    split: str,
    threshold: float,
    model_id: str,
    seed: int,
    run_id: str,
) -> pd.DataFrame:
    probs = validate_probabilities(probs, n_rows=len(frame))
    p_attack = attack_probability(probs)
    out = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str).values if "row_id" in frame.columns else [f"{split}:{i}" for i in range(len(frame))],
            "text_hash": frame["text_hash"].astype(str).values if "text_hash" in frame.columns else None,
            "split": split,
            "gold_label": frame["intent_label"].astype(int).values,
            "gold_label_name": [LABEL_ORDER[int(i)] for i in frame["intent_label"]],
            "pred_label": probs.argmax(axis=1).astype(int),
        }
    )
    out["pred_label_name"] = [LABEL_ORDER[int(i)] for i in out["pred_label"]]
    out["pred_confidence"] = probs.max(axis=1)
    for idx, col in enumerate(PROB_COLUMNS):
        out[col] = probs[:, idx]
    out["binary_gold"] = (out["gold_label"] != 0).astype(int)
    out["binary_prob_attack"] = p_attack
    out["binary_pred"] = (p_attack >= threshold).astype(int)
    out["binary_threshold"] = float(threshold)
    for col in META_COLUMNS:
        out[col] = frame[col].values if col in frame.columns else None
    out["model_id"] = model_id
    out["seed"] = int(seed)
    out["run_id"] = run_id
    return out


def evaluate_run(
    run_dir: str | Path,
    model_id: str,
    seed: int,
    run_id: str,
    val_frame: pd.DataFrame,
    val_probs: np.ndarray,
    test_frame: pd.DataFrame | None = None,
    test_probs: np.ndarray | None = None,
    threshold_objective: str = "f1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the binary threshold on validation, then evaluate validation (and test)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    y_val = val_frame["intent_label"].astype(int).to_numpy()
    selection = select_binary_threshold(y_val, val_probs, objective=threshold_objective)
    threshold = float(selection["threshold"])
    (run_dir / "threshold.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    results: dict[str, Any] = {"threshold": selection}
    bundles = [("validation", val_frame, val_probs)]
    if test_frame is not None and test_probs is not None:
        bundles.append(("test", test_frame, test_probs))
    for split, frame, probs in bundles:
        y = frame["intent_label"].astype(int).to_numpy()
        metrics = evaluate_split(y, probs, threshold)
        metrics["model_id"] = model_id
        metrics["seed"] = int(seed)
        metrics["run_id"] = run_id
        metrics["split"] = split
        if extra:
            metrics["run_info"] = extra
        (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        preds = predictions_frame(frame, probs, split, threshold, model_id, seed, run_id)
        preds.to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
        results[split] = metrics
    return results


def load_predictions(run_dir: str | Path, split: str) -> pd.DataFrame:
    path = Path(run_dir) / f"predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    validate_probabilities(df[PROB_COLUMNS].to_numpy(), n_rows=len(df))
    return df
