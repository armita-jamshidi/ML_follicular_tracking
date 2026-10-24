"""Evaluation metrics. Accuracy alone is never reported: the classes are imbalanced."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import CLASS_NAMES, DISPLAY_NAMES


def classification_metrics(y_true, y_pred, class_names=CLASS_NAMES) -> dict:
    labels = list(range(len(class_names)))
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": {c: float(r) for c, r in zip(class_names, recall)},
        "precision": {c: float(p) for c, p in zip(class_names, precision)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, labels=None) -> tuple[float, float]:
    """95% percentile bootstrap interval for macro-F1 over images."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = labels if labels is not None else list(range(len(CLASS_NAMES)))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize_cv(oof: pd.DataFrame, class_names=CLASS_NAMES) -> dict:
    """Summarize out-of-fold predictions (columns: fold, label_idx, pred, optional acquisition)."""
    per_fold = [classification_metrics(g["label_idx"], g["pred"], class_names) for _, g in oof.groupby("fold")]
    pooled = classification_metrics(oof["label_idx"], oof["pred"], class_names)
    labels = list(range(len(class_names)))
    pooled["macro_f1_95ci"] = bootstrap_ci(oof["label_idx"], oof["pred"], labels=labels)

    def mean_std(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary = {
        "pooled": pooled,
        "per_fold_mean_std": {
            "macro_f1": mean_std([m["macro_f1"] for m in per_fold]),
            "balanced_accuracy": mean_std([m["balanced_accuracy"] for m in per_fold]),
            "recall": {c: mean_std([m["recall"][c] for m in per_fold]) for c in class_names},
        },
        "per_fold": per_fold,
    }
    # Shortcut check: does performance depend on how the image was acquired?
    if "acquisition" in oof:
        summary["by_acquisition"] = {
            acq: {
                "n": int(len(g)),
                "accuracy": float((g["label_idx"] == g["pred"]).mean()),
                "recall": {
                    c: float((g.loc[g["label_idx"] == i, "pred"] == i).mean())
                    for i, c in enumerate(class_names)
                    if (g["label_idx"] == i).any()
                },
            }
            for acq, g in oof.groupby("acquisition")
        }
    return summary


def plot_confusion(cm, class_names=CLASS_NAMES, title: str = "", path: str | Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    names = [DISPLAY_NAMES.get(c, c) for c in class_names]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def save_results(out_dir: str | Path, oof: pd.DataFrame, summary: dict, title: str, class_names=CLASS_NAMES) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_confusion(summary["pooled"]["confusion_matrix"], class_names, title, out_dir / "confusion_matrix.png")


def format_summary(name: str, summary: dict) -> str:
    p, s = summary["pooled"], summary["per_fold_mean_std"]
    lo, hi = p["macro_f1_95ci"]
    recalls = ", ".join(f"{DISPLAY_NAMES.get(c, c)} {v['mean']:.2f}" for c, v in s["recall"].items())
    return (
        f"{name}: macro-F1 {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f} across folds "
        f"(pooled {p['macro_f1']:.3f}, 95% CI {lo:.3f}-{hi:.3f}); recall: {recalls}"
    )
