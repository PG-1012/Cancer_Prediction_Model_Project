"""Evaluation.

Two decisions drive everything here.

**Cross-validation, not a single split.** With 540 samples an 80/20 split
leaves 108 test cases, so one misclassification moves accuracy by almost a
full point. A single number from a single split cannot distinguish a real
difference between models from the luck of `random_state=42`. Repeated
stratified k-fold gives a mean and a spread, and the spread is the part that
matters.

**Recall, not accuracy.** The two errors are not equivalent. A false positive
sends a healthy patient for further testing. A false negative tells someone
with a malignancy that they are clear. Accuracy weighs those the same; recall
and the operating threshold are where the real decision lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .data import Dataset
from .models import RANDOM_STATE

DEFAULT_SPLITS = 5
DEFAULT_REPEATS = 10


@dataclass
class Scores:
    """Cross-validated metrics for one model."""

    name: str
    accuracy: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    roc_auc: np.ndarray
    pr_auc: np.ndarray
    false_negatives: np.ndarray
    false_positives: np.ndarray

    def mean_std(self, metric: str) -> tuple[float, float]:
        values = getattr(self, metric)
        return float(np.mean(values)), float(np.std(values))

    def format(self, metric: str, pct: bool = True) -> str:
        mean, std = self.mean_std(metric)
        scale = 100 if pct else 1
        suffix = "%" if pct else ""
        return f"{mean * scale:.1f}{suffix} ± {std * scale:.1f}"


def _metrics_from(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba),
        "false_negatives": float(fn),
        "false_positives": float(fp),
    }


def cross_validate(
    model,
    data: Dataset,
    splits: int = DEFAULT_SPLITS,
    repeats: int = DEFAULT_REPEATS,
    threshold: float = 0.5,
    name: str = "model",
) -> Scores:
    """Repeated stratified k-fold. Stratified because a fold that happens to
    hold few malignant cases would make recall meaningless."""
    cv = RepeatedStratifiedKFold(
        n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE
    )
    collected: dict[str, list[float]] = {
        k: [] for k in
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
         "false_negatives", "false_positives"]
    }

    for train_idx, test_idx in cv.split(data.X, data.y):
        fitted = clone(model).fit(data.X[train_idx], data.y[train_idx])
        proba = fitted.predict_proba(data.X[test_idx])[:, 1]
        for key, value in _metrics_from(data.y[test_idx], proba, threshold).items():
            collected[key].append(value)

    return Scores(name=name, **{k: np.asarray(v) for k, v in collected.items()})


def out_of_fold_probabilities(
    model, data: Dataset, splits: int = DEFAULT_SPLITS
) -> np.ndarray:
    """Predictions for every sample, each made by a model that never saw it.

    Used for curves and threshold analysis so those are not read off training
    data.
    """
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    proba = np.zeros(len(data.y), dtype=float)
    for train_idx, test_idx in cv.split(data.X, data.y):
        fitted = clone(model).fit(data.X[train_idx], data.y[train_idx])
        proba[test_idx] = fitted.predict_proba(data.X[test_idx])[:, 1]
    return proba


@dataclass
class ThresholdChoice:
    threshold: float
    recall: float
    precision: float
    false_negatives: int
    false_positives: int
    target_recall: float = 0.0


def threshold_for_recall(
    y_true: np.ndarray, proba: np.ndarray, target_recall: float = 0.99
) -> ThresholdChoice:
    """Lowest-cost threshold that still catches `target_recall` of malignancies.

    The default 0.5 cutoff is an arbitrary inheritance from the sigmoid, not a
    clinical choice. If missing a malignancy is the expensive error, the
    threshold should be set from the recall you require and the precision cost
    stated plainly.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one more point than thresholds.
    best: ThresholdChoice | None = None
    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if recall >= target_recall:
            if best is None or precision > best.precision:
                pred = (proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                best = ThresholdChoice(
                    threshold=float(threshold),
                    recall=float(recall),
                    precision=float(precision),
                    false_negatives=int(fn),
                    false_positives=int(fp),
                    target_recall=target_recall,
                )
    if best is None:
        # No threshold reaches the target; report the most sensitive available.
        pred = (proba >= thresholds[0]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        best = ThresholdChoice(
            threshold=float(thresholds[0]),
            recall=float(tp / (tp + fn)) if (tp + fn) else 0.0,
            precision=float(tp / (tp + fp)) if (tp + fp) else 0.0,
            false_negatives=int(fn),
            false_positives=int(fp),
            target_recall=target_recall,
        )
    return best


def curves(y_true: np.ndarray, proba: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision, "recall": recall,
        "pr_auc": average_precision_score(y_true, proba),
    }
