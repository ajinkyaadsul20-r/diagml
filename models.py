"""Candidate model factories. Every model is wrapped in a Pipeline so preprocessing
is fitted only on training folds (no data leakage during cross-validation)."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(name: str, seed: int) -> Pipeline:
    """Build an unfitted pipeline for a named model."""
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=5000, random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: logistic_regression, "
            "random_forest, gradient_boosting"
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
