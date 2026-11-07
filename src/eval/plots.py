"""Publication figures generated ONLY from saved artifacts (predictions, metrics, history).

Every figure function:
* reads ``outputs/runs/**`` (or a smoke directory) artifacts - never recomputes predictions,
* writes ``<name>.png`` (300 DPI) + ``<name>.pdf`` + ``<name>.svg`` with a colorblind-safe palette,
* appends a record to ``skipped.json`` (with a reason) when its prerequisites are missing
  instead of fabricating or silently omitting a figure.

Dataset-only figures (class / source / synthetic distributions) are titled "DATASET" so they
cannot be mistaken for model results.

CLI::

    python -m src.eval.plots --runs outputs/runs --out outputs/figures
    python -m src.eval.plots --runs outputs/smoke --out outputs/smoke/figures --splits-dir data/processed/splits/v1_leakfree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

from src.data.constants import LABEL_ORDER, NUM_CLASSES, SPLITS  # noqa: E402
from src.eval.evaluate import calibration_metrics  # noqa: E402
from src.eval.predict import PROB_COLUMNS, load_predictions  # noqa: E402
from src.eval.tables import collect_run_metrics, discover_runs, group_bootstrap_ci, seed_summary, subgroup_table  # noqa: E402

# Okabe-Ito colorblind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
DPI = 300
FORMATS = ("png", "pdf", "svg")

plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=PALETTE), "font.size": 9, "figure.dpi": 100, "savefig.bbox": "tight"})


class FigureContext:
    def __init__(self, runs_dir: Path, out_dir: Path, split: str = "test", splits_dir: Path | None = None, tag: str = ""):
        self.runs_dir = Path(runs_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.splits_dir = Path(splits_dir) if splits_dir else None
        self.tag = tag
        self.skipped: list[dict[str, str]] = []
        self.written: list[str] = []
        self._run_dirs = discover_runs(self.runs_dir, split)
        self._preds: dict[Path, pd.DataFrame] = {}
        self._metrics: dict[Path, dict] = {}

    # --- data access -------------------------------------------------------
    @property
    def run_dirs(self) -> list[Path]:
        return self._run_dirs

    def preds(self, run_dir: Path) -> pd.DataFrame:
        if run_dir not in self._preds:
            self._preds[run_dir] = load_predictions(run_dir, self.split)
        return self._preds[run_dir]

    def metrics(self, run_dir: Path) -> dict:
        if run_dir not in self._metrics:
            self._metrics[run_dir] = json.loads((run_dir / f"metrics_{self.split}.json").read_text(encoding="utf-8"))
        return self._metrics[run_dir]

    def label(self, run_dir: Path) -> str:
        m = self.metrics(run_dir)
        return f"{m['model_id']} (seed {m['seed']})"

    def best_run_per_model(self) -> dict[str, Path]:
        """Representative run per model = first seed (deterministic), for single-run figures."""
        out: dict[str, Path] = {}
        for rd in self.run_dirs:
            mid = self.metrics(rd)["model_id"]
            out.setdefault(mid, rd)
        return out

    # --- output ------------------------------------------------------------
    def save(self, fig: plt.Figure, name: str) -> None:
        if self.tag:
            fig.suptitle(f"{fig._suptitle.get_text() if fig._suptitle else ''} [{self.tag}]".strip(), fontsize=8, color="gray", y=1.02)
        for ext in FORMATS:
            fig.savefig(self.out_dir / f"{name}.{ext}", dpi=DPI, format=ext)
        plt.close(fig)
        self.written.append(name)

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append({"figure": name, "reason": reason})

    def flush(self) -> dict[str, Any]:
        report = {"runs_dir": str(self.runs_dir.as_posix()), "split": self.split, "written": self.written, "skipped": self.skipped}
        (self.out_dir / "skipped.json").write_text(json.dumps(self.skipped, indent=2), encoding="utf-8")
        (self.out_dir / "figures_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def _cm_axes(ax, cm: np.ndarray, title: str, fmt: str, cmap: str = "Blues") -> None:
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)


# ---------------------------------------------------------------------------
# 1-2 Confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion_counts(ctx: FigureContext) -> None:
    name = "01_confusion_matrix_counts"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        cm = np.array(ctx.metrics(rd)["multiclass"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        _cm_axes(ax, cm, f"Confusion matrix (counts) - {ctx.label(rd)} - {ctx.split}", "d")
        ctx.save(fig, f"{name}_{mid}")


def fig_confusion_normalized(ctx: FigureContext) -> None:
    name = "02_confusion_matrix_normalized"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        cm = np.array(ctx.metrics(rd)["multiclass"]["confusion_matrix_normalized"])
        fig, ax = plt.subplots(figsize=(6, 5))
        _cm_axes(ax, cm, f"Confusion matrix (row-normalised) - {ctx.label(rd)} - {ctx.split}", ".2f")
        ctx.save(fig, f"{name}_{mid}")


# ---------------------------------------------------------------------------
# 3-4 Binary / per-class TP TN FP FN
# ---------------------------------------------------------------------------
def fig_binary_confusion(ctx: FigureContext) -> None:
    name = "03_binary_tp_tn_fp_fn"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        b = ctx.metrics(rd)["binary"]
        cm = np.array(b["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Purples")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred benign", "pred attack"]); ax.set_yticklabels(["gold benign", "gold attack"])
        for (i, j), v in np.ndenumerate(cm):
            tag = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(i, j)]
            ax.text(j, i, f"{tag}\n{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_title(f"Binary attack detection @thr={b['threshold']:.2f} (val-selected)\n{ctx.label(rd)} - {ctx.split}")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ctx.save(fig, f"{name}_{mid}")


def fig_per_class_counts(ctx: FigureContext) -> None:
    name = "04_per_class_ovr_counts"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        counts = ctx.metrics(rd)["multiclass"]["per_class_counts"]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(NUM_CLASSES)
        w = 0.2
        for k, key in enumerate(("tp", "fp", "fn", "tn")):
            ax.bar(x + (k - 1.5) * w, [counts[c][key] for c in LABEL_ORDER], width=w, label=key.upper())
        ax.set_xticks(x); ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right")
        ax.set_yscale("log"); ax.set_ylabel("count (log)")
        ax.set_title(f"Per-class one-vs-rest TP/FP/FN/TN - {ctx.label(rd)} - {ctx.split}")
        ax.legend()
        ctx.save(fig, f"{name}_{mid}")


# ---------------------------------------------------------------------------
# 5 Dataset distribution (dataset only)
# ---------------------------------------------------------------------------
def fig_dataset_distribution(ctx: FigureContext) -> None:
    name = "05_dataset_distribution"
    if ctx.splits_dir is None or not (ctx.splits_dir / "train.parquet").exists():
        return ctx.skip(name, "no --splits-dir with parquet splits")
    frames = {s: pd.read_parquet(ctx.splits_dir / f"{s}.parquet") for s in SPLITS if (ctx.splits_dir / f"{s}.parquet").exists()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(NUM_CLASSES)
    w = 0.27
    for k, (s, df) in enumerate(frames.items()):
        counts = df["intent_label"].astype(int).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        axes[0].bar(x + (k - 1) * w, counts.values, width=w, label=s)
    axes[0].set_xticks(x); axes[0].set_xticklabels(LABEL_ORDER, rotation=30, ha="right"); axes[0].set_title("Intent classes"); axes[0].legend()
    if "source" in frames["train"].columns:
        src = pd.DataFrame({s: df["source"].value_counts() for s, df in frames.items()}).fillna(0)
        src.plot.bar(ax=axes[1], width=0.8); axes[1].set_title("Source categories"); axes[1].tick_params(axis="x", rotation=30)
        syn = {s: (df["source"].astype(str).str.startswith("synthetic")).mean() for s, df in frames.items()}
        axes[2].bar(list(syn), [v * 100 for v in syn.values()]); axes[2].set_ylabel("% synthetic rows"); axes[2].set_title("Synthetic share")
    else:
        axes[1].axis("off"); axes[2].axis("off")
    fig.suptitle(f"DATASET distribution - {ctx.splits_dir.name} (no model results)")
    ctx.save(fig, name)


# ---------------------------------------------------------------------------
# 6-7 Training curves
# ---------------------------------------------------------------------------
def _histories(ctx: FigureContext) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rd in ctx.run_dirs:
        path = rd / "history.jsonl"
        if path.exists():
            out[ctx.label(rd)] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def fig_loss_curves(ctx: FigureContext) -> None:
    name = "06_training_loss_curves"
    hist = _histories(ctx)
    if not hist:
        return ctx.skip(name, "no history.jsonl files")
    fig, ax = plt.subplots(figsize=(7, 4))
    drew = False
    for label, records in hist.items():
        steps = [r for r in records if r.get("type") == "step"]
        epochs = [r for r in records if r.get("type") == "epoch"]
        if steps:
            ax.plot([r["global_step"] for r in steps], [r["loss"] for r in steps], label=label, alpha=0.8); drew = True
        elif epochs:
            ax.plot([r["global_step"] for r in epochs], [r["train_loss"] for r in epochs], marker="o", label=label); drew = True
    if not drew:
        plt.close(fig)
        return ctx.skip(name, "history has no loss records")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("train loss"); ax.set_title("Training loss"); ax.legend(fontsize=7)
    ctx.save(fig, name)


def fig_val_f1_curves(ctx: FigureContext) -> None:
    name = "07_validation_macro_f1_curves"
    hist = _histories(ctx)
    epochs = {k: [r for r in v if r.get("type") == "epoch"] for k, v in hist.items()}
    epochs = {k: v for k, v in epochs.items() if v}
    if not epochs:
        return ctx.skip(name, "no epoch records in history")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, recs in epochs.items():
        ax.plot([r["epoch"] + 1 for r in recs], [r["val_f1_macro"] for r in recs], marker="o", label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1"); ax.set_title("Validation macro-F1 per epoch (model selection only)"); ax.legend(fontsize=7)
    ctx.save(fig, name)


# ---------------------------------------------------------------------------
# 8-9 Cross-model comparisons
# ---------------------------------------------------------------------------
def fig_model_metric_bars(ctx: FigureContext) -> None:
    name = "08_cross_model_metrics"
    table = collect_run_metrics(ctx.runs_dir, ctx.split)
    if table.empty:
        return ctx.skip(name, "no evaluated runs")
    summ = seed_summary(table)
    metrics = [("mc_f1_macro", "macro-F1"), ("mc_accuracy", "accuracy"), ("bin_f1", "binary F1"), ("mc_auroc_ovr_macro", "OvR AUROC")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=False)
    order = summ.sort_values("mc_f1_macro_mean", ascending=False)["model_id"].tolist()
    for ax, (key, title) in zip(axes, metrics):
        means = summ.set_index("model_id").loc[order, f"{key}_mean"]
        stds = summ.set_index("model_id").loc[order, f"{key}_std"].fillna(0)
        ax.barh(order, means.values, xerr=stds.values, color=PALETTE[0], ecolor="black", capsize=3)
        ax.set_title(f"{title} (mean ± std over seeds)"); ax.set_xlim(0, 1); ax.invert_yaxis()
    n_seeds = summ.set_index("model_id")["n_seeds"].to_dict()
    fig.suptitle(f"{ctx.split} - seeds per model: " + ", ".join(f"{m}={n_seeds[m]}" for m in order), fontsize=8)
    ctx.save(fig, name)

    # bootstrap-CI variant (group-aware, first run per model) - separate figure, separate uncertainty
    name_ci = "08b_cross_model_f1_bootstrap_ci"
    cis = {}
    for mid, rd in ctx.best_run_per_model().items():
        ci_path = rd / f"bootstrap_ci_{ctx.split}.json"
        if ci_path.exists():
            cis[mid] = json.loads(ci_path.read_text(encoding="utf-8"))
        else:
            cis[mid] = group_bootstrap_ci(ctx.preds(rd), n_boot=300)
    rows = [(m, c["ci"]["mc_f1_macro"]) for m, c in cis.items() if c["ci"].get("mc_f1_macro", {}).get("low") is not None]
    if not rows:
        return ctx.skip(name_ci, "bootstrap produced too few valid resamples")
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.5))
    for i, (m, ci) in enumerate(rows):
        ax.plot([ci["low"], ci["high"]], [i, i], color=PALETTE[0]); ax.plot(ci["mean"], i, "o", color=PALETTE[3])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1"); ax.set_title(f"95% group-aware bootstrap CI (families) - {ctx.split}, one run per model")
    ctx.save(fig, name_ci)


def fig_model_class_heatmap(ctx: FigureContext) -> None:
    name = "09_model_x_class_f1_heatmap"
    table = collect_run_metrics(ctx.runs_dir, ctx.split)
    if table.empty:
        return ctx.skip(name, "no evaluated runs")
    cols = [f"f1_{c}" for c in LABEL_ORDER]
    mat = table.groupby("model_id")[cols].mean()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(mat) + 2))
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", color="white" if mat.values[i, j] < 0.5 else "black", fontsize=7)
    ax.set_title(f"Per-class F1 (mean over seeds) - {ctx.split}")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ctx.save(fig, name)


# ---------------------------------------------------------------------------
# 10-12 ROC / PR / AUROC-AP
# ---------------------------------------------------------------------------
def fig_roc_curves(ctx: FigureContext) -> None:
    name = "10_roc_ovr"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        df = ctx.preds(rd)
        y = df["gold_label"].to_numpy(int)
        probs = df[PROB_COLUMNS].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        tprs, grid = [], np.linspace(0, 1, 200)
        for k, cls in enumerate(LABEL_ORDER):
            pos = (y == k).astype(int)
            if pos.sum() in (0, len(pos)):
                continue
            fpr, tpr, _ = roc_curve(pos, probs[:, k])
            axes[0].plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
            tprs.append(np.interp(grid, fpr, tpr))
        if not tprs:
            plt.close(fig); ctx.skip(f"{name}_{mid}", "no class with positives and negatives"); continue
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_title(f"One-vs-rest ROC - {ctx.label(rd)}"); axes[0].legend(fontsize=6)
        mean_tpr = np.mean(tprs, axis=0)
        axes[1].plot(grid, mean_tpr, color=PALETTE[0], label=f"macro ROC (AUC={auc(grid, mean_tpr):.3f})")
        yb = (y != 0).astype(int)
        if 0 < yb.sum() < len(yb):
            fpr, tpr, _ = roc_curve(yb, df["binary_prob_attack"])
            axes[1].plot(fpr, tpr, color=PALETTE[1], label=f"binary attack ROC (AUC={auc(fpr, tpr):.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8); axes[1].set_title("Macro-averaged & binary ROC"); axes[1].legend(fontsize=7)
        for ax in axes:
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ctx.save(fig, f"{name}_{mid}")


def fig_pr_curves(ctx: FigureContext) -> None:
    name = "11_precision_recall_ovr"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        df = ctx.preds(rd)
        y = df["gold_label"].to_numpy(int)
        probs = df[PROB_COLUMNS].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        drew = False
        for k, cls in enumerate(LABEL_ORDER):
            pos = (y == k).astype(int)
            if pos.sum() in (0, len(pos)):
                continue
            p, r, _ = precision_recall_curve(pos, probs[:, k])
            prevalence = pos.mean()
            line, = ax.plot(r, p, label=f"{cls} (prev={prevalence:.2f})")
            ax.axhline(prevalence, color=line.get_color(), ls=":", lw=0.6)
            drew = True
        if not drew:
            plt.close(fig); ctx.skip(f"{name}_{mid}", "no class with positives and negatives"); continue
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"OvR precision-recall (dotted = prevalence baseline) - {ctx.label(rd)}"); ax.legend(fontsize=6)
        ctx.save(fig, f"{name}_{mid}")


def fig_auroc_ap_comparison(ctx: FigureContext) -> None:
    name = "12_auroc_ap_comparison"
    table = collect_run_metrics(ctx.runs_dir, ctx.split)
    if table.empty:
        return ctx.skip(name, "no evaluated runs")
    g = table.groupby("model_id")[["mc_auroc_ovr_macro", "mc_ap_ovr_macro", "bin_auroc", "bin_average_precision"]].mean().sort_values("mc_auroc_ovr_macro", ascending=False)
    if g.isna().all().all():
        return ctx.skip(name, "AUROC/AP unavailable for all runs")
    fig, ax = plt.subplots(figsize=(8, 4))
    g.plot.bar(ax=ax, width=0.8); ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.set_title(f"AUROC / AP (macro OvR and binary) - {ctx.split}"); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7)
    ctx.save(fig, name)


# ---------------------------------------------------------------------------
# 13-14 Subsets & ablation
# ---------------------------------------------------------------------------
def fig_subset_performance(ctx: FigureContext) -> None:
    name = "13_subset_performance"
    table = subgroup_table(ctx.runs_dir, ctx.split)
    if table.empty:
        return ctx.skip(name, "no predictions with metadata columns")
    for group_by, part in table.groupby("group_by"):
        piv = part.groupby(["group", "model_id"])["f1_macro"].mean().unstack("model_id")
        ns = part.groupby("group")["n"].max()
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(piv)), 4))
        piv.plot.bar(ax=ax, width=0.8); ax.set_ylim(0, 1); ax.set_ylabel("macro-F1 within subset")
        ax.set_xticklabels([f"{g}\n(n={ns[g]})" for g in piv.index], rotation=0, fontsize=7)
        ax.set_title(f"Performance by {group_by} - {ctx.split} (macro-F1 over classes present in subset)"); ax.legend(fontsize=6)
        ctx.save(fig, f"{name}_{group_by}")


def fig_ablation_training_variants(ctx: FigureContext) -> None:
    name = "14_ablation_train_variants"
    rows = []
    for rd in ctx.run_dirs:
        cfg_path = rd / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        rows.append({"model_id": cfg["model"]["model_id"], "train_file": cfg["data"]["train_file"], "f1_macro": ctx.metrics(rd)["multiclass"]["f1_macro"]})
    df = pd.DataFrame(rows)
    if df.empty or df["train_file"].nunique() < 2:
        return ctx.skip(name, "needs runs trained on >=2 different train variants (clean/balanced/augmented/adversarial) on the same frozen test split")
    piv = df.groupby(["model_id", "train_file"])["f1_macro"].mean().unstack("train_file")
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.plot.bar(ax=ax, width=0.8); ax.set_ylim(0, 1); ax.set_ylabel("test macro-F1"); ax.set_title("Ablation: training-data variant (same frozen test split)"); ax.tick_params(axis="x", rotation=30)
    ctx.save(fig, name)


# ---------------------------------------------------------------------------
# 15-17 Calibration, efficiency, confidence
# ---------------------------------------------------------------------------
def fig_reliability(ctx: FigureContext) -> None:
    name = "15_reliability_diagram"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    fig, ax = plt.subplots(figsize=(5, 5))
    for mid, rd in ctx.best_run_per_model().items():
        df = ctx.preds(rd)
        cal = calibration_metrics(df["gold_label"].to_numpy(int), df[PROB_COLUMNS].to_numpy(), n_bins=10)
        pts = [(b["confidence"], b["accuracy"]) for b in cal["reliability_bins"] if b["count"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{mid} (ECE={cal['ece']:.3f}, Brier={cal['brier_multiclass']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("confidence (top-label)"); ax.set_ylabel("accuracy"); ax.set_title(f"Reliability diagram - {ctx.split}"); ax.legend(fontsize=6)
    ctx.save(fig, name)


def fig_efficiency(ctx: FigureContext) -> None:
    name = "16_efficiency_vs_f1"
    rows = []
    for rd in ctx.run_dirs:
        sp = rd / "run_summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        rows.append({"model_id": s["model_id"], "wall": s.get("train_wall_time_sec"), "params": (s.get("params") or {}).get("params_total"), "f1": ctx.metrics(rd)["multiclass"]["f1_macro"], "gpu": (s.get("hardware") or {}).get("gpu_name", "cpu")})
    df = pd.DataFrame(rows).dropna(subset=["wall", "params"])
    if df.empty:
        return ctx.skip(name, "run_summary.json lacks wall time / parameter counts")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mid, part in df.groupby("model_id"):
        axes[0].scatter(part["wall"], part["f1"], label=mid); axes[1].scatter(part["params"], part["f1"], label=mid)
    axes[0].set_xscale("log"); axes[0].set_xlabel("training wall time (s, log)"); axes[0].set_ylabel("macro-F1")
    axes[1].set_xscale("log"); axes[1].set_xlabel("total parameters (log)"); axes[1].set_ylabel("macro-F1")
    axes[0].legend(fontsize=6)
    hw = ", ".join(sorted(df["gpu"].astype(str).unique()))
    fig.suptitle(f"Efficiency vs macro-F1 - {ctx.split} (hardware: {hw}; peak memory not recorded)")
    ctx.save(fig, name)


def fig_confidence_histograms(ctx: FigureContext) -> None:
    name = "17_confidence_error_histograms"
    if not ctx.run_dirs:
        return ctx.skip(name, "no evaluated runs")
    for mid, rd in ctx.best_run_per_model().items():
        df = ctx.preds(rd)
        correct = df["pred_label"] == df["gold_label"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df.loc[correct, "pred_confidence"], bins=20, alpha=0.7, label=f"correct (n={int(correct.sum())})")
        axes[0].hist(df.loc[~correct, "pred_confidence"], bins=20, alpha=0.7, label=f"errors (n={int((~correct).sum())})")
        axes[0].set_xlabel("top-label confidence"); axes[0].set_ylabel("count"); axes[0].legend(); axes[0].set_title("Confidence distribution")
        err = df[~correct]
        if len(err):
            pairs = err.groupby(["gold_label_name", "pred_label_name"]).size().sort_values(ascending=False).head(12)
            axes[1].barh([f"{g} -> {p}" for g, p in pairs.index], pairs.values, color=PALETTE[3]); axes[1].invert_yaxis(); axes[1].set_title("Most frequent confusions")
        else:
            axes[1].text(0.5, 0.5, "no errors", ha="center"); axes[1].axis("off")
        fig.suptitle(f"{ctx.label(rd)} - {ctx.split}")
        ctx.save(fig, f"{name}_{mid}")


FIGURES: list[Callable[[FigureContext], None]] = [
    fig_confusion_counts,
    fig_confusion_normalized,
    fig_binary_confusion,
    fig_per_class_counts,
    fig_dataset_distribution,
    fig_loss_curves,
    fig_val_f1_curves,
    fig_model_metric_bars,
    fig_model_class_heatmap,
    fig_roc_curves,
    fig_pr_curves,
    fig_auroc_ap_comparison,
    fig_subset_performance,
    fig_ablation_training_variants,
    fig_reliability,
    fig_efficiency,
    fig_confidence_histograms,
]
assert len(FIGURES) == 17


def generate_all(runs_dir: str | Path, out_dir: str | Path, split: str = "test", splits_dir: str | Path | None = None, tag: str = "") -> dict[str, Any]:
    ctx = FigureContext(Path(runs_dir), Path(out_dir), split=split, splits_dir=Path(splits_dir) if splits_dir else None, tag=tag)
    for fn in FIGURES:
        try:
            fn(ctx)
        except Exception as exc:  # noqa: BLE001 - record, never crash the whole batch
            ctx.skip(fn.__name__, f"{type(exc).__name__}: {exc}")
            plt.close("all")
    return ctx.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate publication figures from saved run artifacts.")
    p.add_argument("--runs", default="outputs/runs")
    p.add_argument("--out", default="outputs/figures")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--splits-dir", default="data/processed/splits/v1_leakfree")
    p.add_argument("--tag", default="", help="Watermark, e.g. SMOKE - not for publication")
    args = p.parse_args(argv)
    report = generate_all(args.runs, args.out, args.split, args.splits_dir, args.tag)
    print(f"written: {len(report['written'])}, skipped: {len(report['skipped'])} -> {args.out}")
    for s in report["skipped"]:
        print(f"  skipped {s['figure']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
