"""Run the full evaluation and regenerate every figure and reported number.

    python -m cancer_model.run

Everything in the README comes from here, so the claims stay checkable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plots
from .data import FEATURE_LABELS, load
from .evaluate import (
    cross_validate,
    curves,
    out_of_fold_probabilities,
    threshold_for_recall,
)
from .models import build_models

RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results.json"
PRIMARY_MODEL = "Logistic regression"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10,
                        help="CV repeats (default 10)")
    parser.add_argument("--target-recall", type=float, default=0.99,
                        help="Recall to tune the operating threshold for")
    args = parser.parse_args(argv)

    data = load()
    print(f"Data: {data.summary()}\n")

    models = build_models()
    scores = []
    print(f"{'model':<28} {'accuracy':>14} {'recall':>14} {'precision':>14} {'ROC-AUC':>10}  missed/fold")
    print("-" * 100)
    for name, model in models.items():
        result = cross_validate(model, data, repeats=args.repeats, name=name)
        scores.append(result)
        print(
            f"{name:<28} {result.format('accuracy'):>14} {result.format('recall'):>14} "
            f"{result.format('precision'):>14} "
            f"{result.mean_std('roc_auc')[0]:>10.3f}  "
            f"{result.mean_std('false_negatives')[0]:.2f}"
        )

    # Curves and threshold analysis use out-of-fold predictions so nothing is
    # read off training data.
    primary = models[PRIMARY_MODEL]
    proba = out_of_fold_probabilities(primary, data)
    curve_data = curves(data.y, proba)
    tuned = threshold_for_recall(data.y, proba, args.target_recall)

    default_pred = (proba >= 0.5).astype(int)
    default_fn = int(((data.y == 1) & (default_pred == 0)).sum())
    default_fp = int(((data.y == 0) & (default_pred == 1)).sum())

    # Threshold choice turns out to matter more than model choice, so show it
    # for every serious candidate rather than only the primary.
    print(f"\nOperating point, out-of-fold predictions ({data.n_malignant} malignant cases)")
    print(f"{'model':<28} {'missed @0.50':>13} {'tuned':>7} {'missed':>8} {'false alarms':>14}")
    print("-" * 78)
    operating = {}
    for name in ["Logistic regression", "Random forest", "MLP (scaled, 300 epochs)"]:
        p_oof = out_of_fold_probabilities(models[name], data)
        pred = (p_oof >= 0.5).astype(int)
        fn = int(((data.y == 1) & (pred == 0)).sum())
        fp = int(((data.y == 0) & (pred == 1)).sum())
        choice = threshold_for_recall(data.y, p_oof, args.target_recall)
        operating[name] = {
            "default": {"threshold": 0.5, "false_negatives": fn, "false_positives": fp},
            "tuned": {
                "threshold": choice.threshold, "recall": choice.recall,
                "precision": choice.precision,
                "false_negatives": choice.false_negatives,
                "false_positives": choice.false_positives,
            },
        }
        print(f"{name:<28} {fn:>13} {choice.threshold:>7.2f} "
              f"{choice.false_negatives:>8} {choice.false_positives:>14}")

    # Figures
    plots.class_distribution(data)
    plots.model_comparison(scores, "recall")
    plots.model_comparison(scores, "accuracy")
    plots.roc_and_pr(curve_data, f"{PRIMARY_MODEL}, out-of-fold predictions")
    plots.confusion_pair(data.y, proba, 0.5, tuned.threshold)
    plots.threshold_sweep(data.y, proba)

    fitted = primary.fit(data.X, data.y)
    coefficients = fitted.named_steps["clf"].coef_[0]
    plots.feature_importance(
        [FEATURE_LABELS[n] for n in data.feature_names],
        coefficients,
        "What the model keys on (logistic regression, standardised)",
    )
    print(f"\nFigures written to {plots.FIG_DIR}")

    payload = {
        "dataset": {
            "samples": len(data), "features": len(data.feature_names),
            "benign": data.n_benign, "malignant": data.n_malignant,
        },
        "cv": {"splits": 5, "repeats": args.repeats},
        "models": {
            s.name: {
                m: {"mean": s.mean_std(m)[0], "std": s.mean_std(m)[1]}
                for m in ["accuracy", "precision", "recall", "f1", "roc_auc",
                          "pr_auc", "false_negatives", "false_positives"]
            }
            for s in scores
        },
        "target_recall": args.target_recall,
        "operating_points": operating,
        "primary_model": PRIMARY_MODEL,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    main()
