"""Loading and preparing the Wisconsin breast cytology data.

Nine cytological attributes scored 1-10 by a pathologist from a fine-needle
aspirate slide, and a binary label. Note what the inputs are: these are
*already-collected biopsy measurements*, not patient risk factors. The model
supports a diagnosis that is already underway; it does not predict whether a
person will develop cancer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "breast_cancer_wisconsin.csv"

TARGET = "Class"

#: Feature names as they appear in the CSV, with readable labels for plots.
FEATURE_LABELS = {
    "Cl.thickness": "Clump thickness",
    "Cell.size": "Uniformity of cell size",
    "Cell.shape": "Uniformity of cell shape",
    "Marg.adhesion": "Marginal adhesion",
    "Epith.c.size": "Single epithelial cell size",
    "Bare.nuclei": "Bare nuclei",
    "Bl.cromatin": "Bland chromatin",
    "Normal.nucleoli": "Normal nucleoli",
    "Mitoses": "Mitoses",
}

FEATURES = list(FEATURE_LABELS)

#: 0 = benign, 1 = malignant.
CLASS_NAMES = {0: "benign", 1: "malignant"}


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_malignant(self) -> int:
        return int(self.y.sum())

    @property
    def n_benign(self) -> int:
        return int(len(self.y) - self.y.sum())

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def summary(self) -> str:
        return (
            f"{len(self)} samples, {len(self.feature_names)} features | "
            f"{self.n_benign} benign, {self.n_malignant} malignant "
            f"({self.prevalence:.1%} malignant)"
        )


def load(path: Path | str = DEFAULT_DATA) -> Dataset:
    """Read the CSV and validate it before anything downstream trusts it."""
    frame = pd.read_csv(path)
    return from_frame(frame)


def from_frame(frame: pd.DataFrame) -> Dataset:
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(
            "Data contains missing values. The canonical Wisconsin dataset encodes "
            "16 unknown 'Bare.nuclei' readings as '?'; decide explicitly how to "
            "handle them rather than letting them through."
        )

    labels = set(np.unique(frame[TARGET]))
    if not labels <= {0, 1}:
        raise ValueError(f"Class column must be 0/1, found {sorted(labels)}")

    out_of_range = [
        name for name in FEATURES
        if frame[name].min() < 1 or frame[name].max() > 10
    ]
    if out_of_range:
        raise ValueError(f"Features must be scored 1-10; out of range: {out_of_range}")

    return Dataset(
        X=frame[FEATURES].to_numpy(dtype=np.float64),
        y=frame[TARGET].to_numpy(dtype=np.int64),
        feature_names=list(FEATURES),
    )
