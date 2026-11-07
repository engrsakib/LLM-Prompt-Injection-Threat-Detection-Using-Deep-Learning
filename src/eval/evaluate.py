"""Comprehensive classification metrics for the 7-class intent task plus the derived binary task.

All functions operate on numpy arrays:

* ``y_true``: int array of shape (N,) with values in ``range(NUM_CLASSES)``
* ``probs``:  float array of shape (N, NUM_CLASSES), finite, rows summing to 1 (validated)

The binary task is derived deterministically: ``p_attack = probs[:, 1:].sum(axis=1)`` and the
decision threshold is **always** chosen on the validation split (``select_binary_threshold``),
never on test. Metrics that cannot be computed (e.g. AUROC when a class is missing in the gold
labels) are reported as ``None`` together with a reason in ``metrics["unavailable"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.constants import INTENT_LABELS, LABEL_ORDER, NUM_CLASSES

DEFAULT_BINARY_THRESHOLD = 0.5


class ProbabilityValidationError(ValueError):
    """Raised when a probability matrix has the wrong shape / content."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_probabilities(probs: np.ndarray, n_rows: int | None = None, atol: float = 1e-3) -> np.ndarray:
    """Return ``probs`` as float64 after validating shape (N, NUM_CLASSES), finiteness, row sums."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != NUM_CLASSES:
        raise ProbabilityValidationError(f"probabilities must have shape (N, {NUM_CLASSES}); got {arr.shape}")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ProbabilityValidationError(f"probabilities have {arr.shape[0]} rows but {n_rows} labels were given")
    if arr.shape[0] == 0:
        raise ProbabilityValidationError("probabilities are empty")
    if not np.all(np.isfinite(arr)):
        raise ProbabilityValidationError("probabilities contain NaN/inf")
    if np.any(arr < -atol) or np.any(arr > 1 + atol):
        raise ProbabilityValidationError("probabilities outside [0, 1]")
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol):
        bad = int(np.sum(~np.isclose(sums, 1.0, atol=atol)))
        raise ProbabilityValidationError(f"{bad} probability rows do not sum to 1 (atol={atol})")
    return arr


def validate_labels(y_true: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    arr = np.asarray(y_true).astype(int)
    if arr.ndim != 1:
        raise ProbabilityValidationError("labels must be 1-D")
    if n_rows is not None and arr.shape[0] != n_rows:
        raise ProbabilityValidationError("labels/probabilities length mismatch")
    if arr.size and (arr.min() < 0 or arr.max() >= NUM_CLASSES):
        raise ProbabilityValidationError(f"labels must be in [0, {NUM_CLASSES - 1}]")
    return arr


# ---------------------------------------------------------------------------
# Binary derivation
# ---------------------------------------------------------------------------
def attack_probability(probs: np.ndarray) -> np.ndarray:
    """p(attack) = sum of all non-benign class probabilities (classes 1..6)."""
    arr = validate_probabilities(probs)
    return np.clip(arr[:, 1:].sum(axis=1), 0.0, 1.0)


def binary_truth(y_true: np.ndarray) -> np.ndarray:
    return (validate_labels(y_true) != 0).astype(int)


def select_binary_threshold(
    y_true_val: np.ndarray,
    probs_val: np.ndarray,
    grid: np.ndarray | None = None,
    objective: str = "f1",
) -> dict[str, Any]:
    """Choose the binary threshold on the VALIDATION split maximising ``objective`` (f1 | balanced_accuracy)."""
    y_bin = binary_truth(y_true_val)
    p_attack = attack_probability(probs_val)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    if len(np.unique(y_bin)) < 2:
        return {
            "threshold": DEFAULT_BINARY_THRESHOLD,
            "objective": objective,
            "score": None,
            "reason": "validation split has a single binary class; default threshold used",
            "selected_on": "validation",
        }
    best_t, best_s = DEFAULT_BINARY_THRESHOLD, -1.0
    for t in grid:
        pred = (p_attack >= t).astype(int)
        if objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_bin, pred)
        else:
            score = f1_score(y_bin, pred, zero_division=0)
        if score > best_s + 1e-12:
            best_t, best_s = float(t), float(score)
    return {"threshold": best_t, "objective": objective, "score": best_s, "selected_on": "validation"}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _per_class_counts(cm: np.ndarray) -> dict[str, dict[str, int]]:
    total = int(cm.sum())
    out: dict[str, dict[str, int]] = {}
    for idx, name in enumerate(LABEL_ORDER):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        tn = total - tp - fp - fn
        out[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": int(cm[idx, :].sum())}
    return out


def _safe(fn, *args, unavailable: dict[str, str], key: str, **kwargs):
    try:
        value = fn(*args, **kwargs)
        return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value)
    except ValueError as exc:  # e.g. only one class present
        unavailable[key] = str(exc)
        return None


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Top-label expected calibration error (equal-width bins), multiclass Brier score, NLL, reliability bins."""
    y_true = validate_labels(y_true)
    probs = validate_probabilities(probs, n_rows=len(y_true))
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"low": float(lo), "high": float(hi), "count": 0, "confidence": None, "accuracy": None})
            continue
        avg_conf, avg_acc = float(conf[mask].mean()), float(correct[mask].mean())
        ece += (count / len(conf)) * abs(avg_acc - avg_conf)
        bins.append({"low": float(lo), "high": float(hi), "count": count, "confidence": avg_conf, "accuracy": avg_acc})
    onehot = np.eye(NUM_CLASSES)[y_true]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))
    return {"ece": float(ece), "n_bins": n_bins, "brier_multiclass": brier, "nll": nll, "mean_confidence": float(conf.mean()), "reliability_bins": bins}


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_true = validate_labels(y_true)
    probs = validate_probabilities(probs, n_rows=len(y_true))
    y_pred = probs.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    unavailable: dict[str, str] = {}

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "label_order": list(LABEL_ORDER),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred, unavailable=unavailable, key="mcc"),
        "per_class": {
            INTENT_LABELS[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
                "present_in_gold": bool(s[i] > 0),
            }
            for i in labels
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class_counts": _per_class_counts(cm),
        "classes_missing_in_gold": [INTENT_LABELS[i] for i in labels if s[i] == 0],
    }
    for avg in ("macro", "micro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(pa)
        metrics[f"recall_{avg}"] = float(ra)
        metrics[f"f1_{avg}"] = float(fa)

    # One-vs-rest AUROC / AP: need both positives and negatives for each class
    y_onehot = np.eye(NUM_CLASSES)[y_true]
    present = [i for i in labels if 0 < s[i] < len(y_true)]
    per_class_auroc: dict[str, float | None] = {}
    per_class_ap: dict[str, float | None] = {}
    for i in labels:
        name = INTENT_LABELS[i]
        if i in present:
            per_class_auroc[name] = float(roc_auc_score(y_onehot[:, i], probs[:, i]))
            per_class_ap[name] = float(average_precision_score(y_onehot[:, i], probs[:, i]))
        else:
            per_class_auroc[name] = None
            per_class_ap[name] = None
            unavailable[f"auroc_ovr[{name}]"] = "class absent (or exclusively present) in gold labels"
    metrics["auroc_ovr_per_class"] = per_class_auroc
    metrics["ap_ovr_per_class"] = per_class_ap
    if present:
        metrics["auroc_ovr_macro"] = float(np.mean([per_class_auroc[INTENT_LABELS[i]] for i in present]))
        metrics["ap_ovr_macro"] = float(np.mean([per_class_ap[INTENT_LABELS[i]] for i in present]))
        metrics["auroc_ovr_macro_classes_used"] = len(present)
    else:
        metrics["auroc_ovr_macro"] = None
        metrics["ap_ovr_macro"] = None
        unavailable["auroc_ovr_macro"] = "no class with both positives and negatives"
    if present:
        weights = np.array([s[i] for i in present], dtype=float)
        metrics["auroc_ovr_weighted"] = float(np.average([per_class_auroc[INTENT_LABELS[i]] for i in present], weights=weights))
    else:
        metrics["auroc_ovr_weighted"] = None
    metrics["calibration"] = calibration_metrics(y_true, probs)
    metrics["unavailable"] = unavailable
    return metrics


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_bin = binary_truth(y_true)
    p_attack = attack_probability(probs)
    pred = (p_attack >= threshold).astype(int)
    unavailable: dict[str, str] = {}
    cm = confusion_matrix(y_bin, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    p, r, f, _ = precision_recall_fscore_support(y_bin, pred, labels=[1], average="binary", zero_division=0)
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "threshold_source": "validation",
        "n": int(len(y_bin)),
        "accuracy": float(accuracy_score(y_bin, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_bin, pred)) if len(np.unique(y_bin)) > 1 else None,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "mcc": _safe(matthews_corrcoef, y_bin, pred, unavailable=unavailable, key="mcc"),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "confusion_matrix": cm.astype(int).tolist(),
    }
    if len(np.unique(y_bin)) > 1:
        out["auroc"] = float(roc_auc_score(y_bin, p_attack))
        out["average_precision"] = float(average_precision_score(y_bin, p_attack))
    else:
        out["auroc"] = None
        out["average_precision"] = None
        unavailable["auroc"] = "single binary class in gold labels"
    out["unavailable"] = unavailable
    return out


def evaluate_split(y_true: np.ndarray, probs: np.ndarray, binary_threshold: float) -> dict[str, Any]:
    """Full metric bundle for one split. ``binary_threshold`` must come from validation."""
    return {
        "multiclass": multiclass_metrics(y_true, probs),
        "binary": binary_metrics(y_true, probs, binary_threshold),
    }


# Backwards-compatible tiny helper used by older code/tests.
def classification_report(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f),
        "precision_macro": float(p),
        "recall_macro": float(r),
    }


__all__ = [
    "ProbabilityValidationError",
    "attack_probability",
    "binary_metrics",
    "binary_truth",
    "calibration_metrics",
    "classification_report",
    "evaluate_split",
    "multiclass_metrics",
    "select_binary_threshold",
    "validate_labels",
    "validate_probabilities",
]
