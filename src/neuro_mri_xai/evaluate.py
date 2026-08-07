"""Evaluation metrics, confusion matrix, ROC curves, and sklearn baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from neuro_mri_xai.config import load_config
from neuro_mri_xai.dataset import get_dataloaders
from neuro_mri_xai.explainability import load_checkpoint_model
from neuro_mri_xai.models.sam_roi import make_roi_fn, unload_sam
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_confusion_matrix
from neuro_mri_xai.utils.seed import set_seed


@torch.no_grad()
def evaluate_classifier(model, loader, class_names, device) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }
    try:
        metrics["auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auc_macro"] = None
    return {"metrics": metrics, "confusion_matrix": confusion_matrix(y_true, y_pred),
            "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def plot_roc_curves(y_true, y_prob, class_names, output_path: Path) -> None:
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
def extract_embeddings(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    inner = getattr(model, "base_model", model)
    if hasattr(inner, "head"):
        inner.head = torch.nn.Identity()
    features, labels = [], []
    for images, lbls in loader:
        feat = inner(images.to(device))
        if feat.dim() > 2:
            feat = feat.mean(dim=[2, 3]) if feat.dim() == 4 else feat.mean(dim=1)
        features.append(feat.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def run_sklearn_baselines(X, y, seed=42) -> dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    models = {
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
    }
    return {name: float(accuracy_score(y_test, clf.fit(X_train, y_train).predict(X_test)))
            for name, clf in models.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swin MRI classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    set_seed(config.dataset.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_fn = make_roi_fn(config) if config.sam.enabled else None
    _, _, test_loader, _ = get_dataloaders(config, roi_fn=roi_fn)
    if config.sam.enabled:
        unload_sam()
    model, class_names = load_checkpoint_model(args.checkpoint, config)
    results = evaluate_classifier(model, test_loader, class_names, device)
    figures_dir = ensure_dir(config.evaluation.figures_dir)
    save_confusion_matrix(results["confusion_matrix"], class_names, figures_dir / "confusion_matrix.png")
    plot_roc_curves(results["y_true"], results["y_prob"], class_names, figures_dir / "roc_curves.png")
    payload = {k: v for k, v in results["metrics"].items() if k != "classification_report"}
    with open(figures_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(figures_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(results["metrics"]["classification_report"])
    print("Test metrics:", payload)
    print(results["metrics"]["classification_report"])
    if config.evaluation.run_sklearn_baselines:
        _, _, test_loader_emb, _ = get_dataloaders(config, roi_fn=roi_fn)
        X, y = extract_embeddings(model, test_loader_emb, device)
        baseline_results = run_sklearn_baselines(X, y, seed=config.dataset.seed)
        print("Sklearn baselines on Swin embeddings:", baseline_results)
        with open(figures_dir / "sklearn_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=2)


if __name__ == "__main__":
    main()
