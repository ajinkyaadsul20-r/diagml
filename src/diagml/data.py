"""Data loading and splitting. Uses the dataset bundled with scikit-learn (no network needed)."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "target"
# In this dataset 0 = malignant, 1 = benign.
CLASS_NAMES = {0: "malignant", 1: "benign"}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and target vector."""
    bunch = load_breast_cancer(as_frame=True)
    X: pd.DataFrame = bunch.data
    y: pd.Series = bunch.target.rename(TARGET)
    return X, y


def feature_names() -> list[str]:
    return list(load_data()[0].columns)


def data_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Stable hash of the dataset, stored with the model for provenance."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()[:16]


def split(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
