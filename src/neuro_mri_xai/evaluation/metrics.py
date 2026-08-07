# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Classification metrics: Accuracy, Precision, Recall, F1, and AUC-ROC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from neuro_mri_xai.utils.paths import ensure_dir


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0,
        ),
    }
    try:
        metrics["auc_macro"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
        )
    except ValueError:
        metrics["auc_macro"] = None
    return metrics


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    return {
        "metrics": compute_metrics(y_true, y_pred, y_prob, class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc = roc_auc_score(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    inner = getattr(model, "base_model", model)
    if hasattr(inner, "head"):
        inner.head = torch.nn.Identity()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, lbls in loader:
        feat = inner(images.to(device))
        if feat.dim() > 2:
            feat = feat.mean(dim=[2, 3]) if feat.dim() == 4 else feat.mean(dim=1)
        features.append(feat.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def run_sklearn_baselines(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    models = {
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
    }
    return {
        name: float(accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test)))
        for name, clf in models.items()
    }
