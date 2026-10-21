"""Metrics and diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute classification metrics from positive-class probabilities."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(
    y_true: Any,
    proba: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC on the test set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(roc_auc_score(y[idx], proba[idx]))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def save_plots(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba: np.ndarray,
    out_dir: Path,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    """Write ROC, confusion matrix, calibration and permutation-importance plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = (proba >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="best model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["malignant", "benign"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)

    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=seed, n_jobs=1
    )
    imp = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    top = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3b6fb6")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title("Top 10 permutation importances (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    return {"top_features": imp.head(5)["feature"].tolist()}
