"""Tests for the data contract and the evaluation logic.

The model's own accuracy is not asserted here — that is measured, not
promised. What is pinned is everything that could silently corrupt a
measurement: validation that rejects bad data, folds that keep both classes,
and threshold selection that actually hits its recall target.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cancer_model.data import FEATURES, TARGET, Dataset, from_frame, load
from cancer_model.evaluate import (
    cross_validate,
    out_of_fold_probabilities,
    threshold_for_recall,
)
from cancer_model.models import build_models


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {name: rng.integers(1, 11, n) for name in FEATURES}
    data[TARGET] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


class TestDataValidation:
    def test_loads_the_shipped_dataset(self):
        data = load()
        assert len(data) == 540
        assert data.X.shape == (540, 9)
        assert set(np.unique(data.y)) == {0, 1}

    def test_counts_are_consistent(self):
        data = load()
        assert data.n_benign + data.n_malignant == len(data)
        assert 0 < data.prevalence < 1

    def test_rejects_missing_columns(self):
        frame = _frame().drop(columns=["Mitoses"])
        with pytest.raises(ValueError, match="Missing expected columns"):
            from_frame(frame)

    def test_rejects_missing_values(self):
        """The canonical Wisconsin data encodes 16 unknown readings as '?'.
        Letting NaNs through would silently drop or corrupt rows."""
        frame = _frame()
        frame.loc[0, "Bare.nuclei"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            from_frame(frame)

    def test_rejects_non_binary_labels(self):
        frame = _frame()
        frame.loc[0, TARGET] = 4          # the raw UCI encoding uses 2 and 4
        with pytest.raises(ValueError, match="0/1"):
            from_frame(frame)

    def test_rejects_out_of_range_features(self):
        frame = _frame()
        frame.loc[0, "Cell.size"] = 99
        with pytest.raises(ValueError, match="1-10"):
            from_frame(frame)


class TestEvaluation:
    @pytest.fixture(scope="class")
    def data(self):
        return load()

    def test_majority_baseline_never_finds_a_malignancy(self, data):
        """A model predicting the common class scores ~56% accuracy here.
        Any headline accuracy has to be read against that floor."""
        scores = cross_validate(
            build_models()["Majority class"], data, repeats=1, name="baseline"
        )
        assert scores.mean_std("recall")[0] == 0.0
        assert 0.5 < scores.mean_std("accuracy")[0] < 0.6

    def test_logistic_regression_beats_the_baseline(self, data):
        scores = cross_validate(
            build_models()["Logistic regression"], data, repeats=1, name="lr"
        )
        assert scores.mean_std("recall")[0] > 0.9
        assert scores.mean_std("roc_auc")[0] > 0.95

    def test_cross_validation_reports_spread(self, data):
        scores = cross_validate(
            build_models()["Logistic regression"], data, repeats=2, name="lr"
        )
        assert len(scores.accuracy) == 10          # 5 folds × 2 repeats
        assert scores.mean_std("accuracy")[1] > 0  # a single split hides this

    def test_out_of_fold_covers_every_sample_once(self, data):
        proba = out_of_fold_probabilities(build_models()["Logistic regression"], data)
        assert len(proba) == len(data)
        assert ((proba >= 0) & (proba <= 1)).all()


class TestThresholdSelection:
    def test_tuned_threshold_meets_its_recall_target(self):
        data = load()
        proba = out_of_fold_probabilities(build_models()["Logistic regression"], data)
        choice = threshold_for_recall(data.y, proba, target_recall=0.99)
        assert choice.recall >= 0.99
        assert 0 < choice.threshold < 1

    def test_tuning_for_recall_reduces_missed_malignancies(self):
        """The central claim of the project: the default 0.5 cutoff is an
        arbitrary inheritance from the sigmoid, and moving it catches cases
        the model already ranked correctly."""
        data = load()
        proba = out_of_fold_probabilities(build_models()["Logistic regression"], data)
        default_fn = int(((data.y == 1) & (proba < 0.5)).sum())
        tuned = threshold_for_recall(data.y, proba, target_recall=0.99)
        assert tuned.false_negatives < default_fn

    def test_higher_target_never_misses_more(self):
        data = load()
        proba = out_of_fold_probabilities(build_models()["Logistic regression"], data)
        lenient = threshold_for_recall(data.y, proba, target_recall=0.90)
        strict = threshold_for_recall(data.y, proba, target_recall=0.99)
        assert strict.false_negatives <= lenient.false_negatives


class TestModels:
    def test_every_model_exposes_probabilities(self):
        # Take rows from both classes. The dataset is grouped by class, so a
        # naive head slice is entirely benign -- and a single-class fit makes
        # DummyClassifier return one probability column, which would break
        # the [:, 1] indexing every metric relies on. Stratified CV is what
        # keeps that from happening for real.
        full = load()
        benign = np.flatnonzero(full.y == 0)[:40]
        malignant = np.flatnonzero(full.y == 1)[:40]
        idx = np.concatenate([benign, malignant])
        data = Dataset(X=full.X[idx], y=full.y[idx], feature_names=full.feature_names)

        for name, model in build_models().items():
            fitted = model.fit(data.X, data.y)
            proba = fitted.predict_proba(data.X)
            assert proba.shape == (len(data), 2), name
            assert np.allclose(proba.sum(axis=1), 1), name
