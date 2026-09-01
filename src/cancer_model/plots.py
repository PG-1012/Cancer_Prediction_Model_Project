"""Figures for the README.

Deliberately plain: no gradients, no 3D, no chartjunk. Each figure answers one
question, and the colour carries meaning rather than decoration -- red is
always the costly error (a missed malignancy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .data import CLASS_NAMES

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

BENIGN = "#2E7D8A"
MALIGNANT = "#C0392B"
NEUTRAL = "#7F8C8D"
ACCENT = "#2C3E50"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def model_comparison(scores: list, metric: str = "recall") -> Path:
    """Mean ± std across folds. The error bars are the point: they show
    whether the gaps between models are real or noise."""
    names = [s.name for s in scores]
    means = [s.mean_std(metric)[0] * 100 for s in scores]
    stds = [s.mean_std(metric)[1] * 100 for s in scores]
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(names))
    ax.barh(
        y, [means[i] for i in order], xerr=[stds[i] for i in order],
        color=[NEUTRAL if "Majority" in names[i] else BENIGN for i in order],
        error_kw={"ecolor": ACCENT, "capsize": 3, "lw": 1},
        height=0.62,
    )
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%), 5-fold CV × 10 repeats")
    ax.set_xlim(0, 104)
    for i, idx in enumerate(order):
        ax.text(means[idx] + stds[idx] + 1.2, i, f"{means[idx]:.1f}",
                va="center", fontsize=8, color=ACCENT)
    ax.set_title(f"{metric.replace('_', ' ').title()} by model", loc="left", fontweight="bold")
    return _save(fig, f"model_comparison_{metric}.png")


def roc_and_pr(curve_data: dict, title: str) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    ax1.plot(curve_data["fpr"], curve_data["tpr"], color=BENIGN, lw=2)
    ax1.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC — AUC {curve_data['roc_auc']:.3f}", loc="left", fontweight="bold")

    ax2.plot(curve_data["recall"], curve_data["precision"], color=MALIGNANT, lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(f"Precision–Recall — AP {curve_data['pr_auc']:.3f}",
                  loc="left", fontweight="bold")

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "roc_pr_curves.png")


def confusion_pair(y_true, proba, default_t: float, tuned_t: float) -> Path:
    """Default cutoff beside the recall-tuned one, so the trade is visible."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, threshold, label in zip(
        axes, [default_t, tuned_t],
        [f"Default threshold ({default_t:.2f})", f"Recall-tuned ({tuned_t:.2f})"],
    ):
        pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for (i, j), value in np.ndenumerate(cm):
            costly = (i == 1 and j == 0)  # actual malignant, predicted benign
            ax.text(
                j, i, f"{value}", ha="center", va="center",
                fontsize=15, fontweight="bold",
                color=MALIGNANT if costly else ("white" if value > cm.max() * 0.5 else ACCENT),
            )
        ax.set_xticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_yticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(False)
    fig.suptitle("Missed malignancies shown in red", x=0.01, ha="left", fontsize=9,
                 color=MALIGNANT)
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


def threshold_sweep(y_true, proba) -> Path:
    """Precision and recall across every cutoff, with the count of missed
    malignancies on a second axis."""
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, misses = [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precisions.append(tp / (tp + fp) if (tp + fp) else 1.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        misses.append(fn)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(thresholds, np.array(recalls) * 100, color=MALIGNANT, lw=2, label="Recall")
    ax.plot(thresholds, np.array(precisions) * 100, color=BENIGN, lw=2, label="Precision")
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=1)
    ax.text(0.51, 40, "default 0.5", fontsize=8, color=NEUTRAL, rotation=90)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Percent")
    ax.set_ylim(40, 102)
    ax.legend(loc="lower left", frameon=False)

    ax2 = ax.twinx()
    ax2.plot(thresholds, misses, color=ACCENT, lw=1, alpha=0.45)
    ax2.set_ylabel("Missed malignancies", color=ACCENT)
    ax2.grid(False)

    ax.set_title("Choosing the operating point", loc="left", fontweight="bold")
    return _save(fig, "threshold_sweep.png")


def feature_importance(names: list[str], values: np.ndarray, title: str) -> Path:
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.barh(np.arange(len(names)), values[order],
            color=[MALIGNANT if v > 0 else BENIGN for v in values[order]], height=0.62)
    ax.set_yticks(np.arange(len(names)), [names[i] for i in order])
    ax.axvline(0, color=ACCENT, lw=0.8)
    ax.set_xlabel("Coefficient (higher → more indicative of malignancy)")
    ax.set_title(title, loc="left", fontweight="bold")
    return _save(fig, "feature_importance.png")


def class_distribution(data) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    counts = [data.n_benign, data.n_malignant]
    ax.bar(["benign", "malignant"], counts, color=[BENIGN, MALIGNANT], width=0.55)
    for i, c in enumerate(counts):
        ax.text(i, c + 6, str(c), ha="center", fontsize=9, color=ACCENT)
    ax.set_ylabel("Samples")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title("Class balance", loc="left", fontweight="bold")
    return _save(fig, "class_distribution.png")
