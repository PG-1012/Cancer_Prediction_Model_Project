"""Candidate classifiers, all behind one scikit-learn-compatible interface.

The original coursework model was a small PyTorch MLP. It is kept here --
wrapped so it can be cross-validated on equal terms -- because the point of
this repo is the comparison: on 540 samples and nine ordinal features, does a
neural network actually earn its place against a linear model?
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

RANDOM_STATE = 42


class TorchMLP(nn.Module):
    """The original architecture: 9 -> 10 -> ReLU -> 1 logit."""

    def __init__(self, n_features: int, hidden: int = 10):
        super().__init__()
        self.layer_1 = nn.Linear(n_features, hidden)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.relu(self.layer_1(x)))


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn wrapper so the MLP can go through the same CV as everything else.

    The original notebook trained with unscaled inputs and plain SGD for 100
    epochs. Both are kept configurable so the effect of fixing them is
    measurable rather than assumed.
    """

    def __init__(
        self,
        hidden: int = 10,
        epochs: int = 300,
        lr: float = 0.1,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        self.classes_ = np.array([0, 1])
        self.model_ = TorchMLP(X_t.shape[1], self.hidden)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for _ in range(self.epochs):
            logits = self.model_(X_t).squeeze(-1)
            loss = loss_fn(logits, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.inference_mode():
            p = torch.sigmoid(self.model_(X_t).squeeze(-1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models() -> dict[str, object]:
    """Every candidate, each scaled inside its own pipeline.

    Scaling belongs in the pipeline, not applied to the whole dataset up
    front: fitting a scaler on data that later becomes the validation fold
    leaks information and quietly inflates every score.
    """
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP (original, unscaled)": TorchMLPClassifier(epochs=100),
        "MLP (scaled, 300 epochs)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", TorchMLPClassifier(epochs=300)),
        ]),
    }
