"""Training: model comparison with cross-validated hyper-parameter search, held-out
evaluation, and persistence of the winning model with full provenance metadata."""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import __version__
from .config import Config
from .data import data_fingerprint, load_data, split
from .evaluate import bootstrap_ci, compute_metrics, save_plots
from .models import build_pipeline

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def _search_space_size(params: dict[str, list[Any]]) -> int:
    size = 1
    for values in params.values():
        size *= len(values)
    return size


def train(cfg: Config) -> dict[str, Any]:
    """Run the full training workflow and return the metadata dict that was saved."""
    started = time.perf_counter()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y, cfg.test_size, cfg.seed)
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    leaderboard: list[dict[str, Any]] = []
    best_name, best_search = "", None
    for name, spec in cfg.models.items():
        n_iter = min(cfg.n_iter, _search_space_size(spec.params)) if spec.params else 1
        search = RandomizedSearchCV(
            build_pipeline(name, cfg.seed),
            param_distributions=spec.params or {"scaler__with_mean": [True]},
            n_iter=n_iter,
            scoring=cfg.scoring,
            cv=cv,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
            refit=True,
        )
        search.fit(X_train, y_train)
        idx = search.best_index_
        row = {
            "model": name,
            f"cv_{cfg.scoring}_mean": float(search.cv_results_["mean_test_score"][idx]),
            f"cv_{cfg.scoring}_std": float(search.cv_results_["std_test_score"][idx]),
            "best_params": {k: _py(v) for k, v in search.best_params_.items()},
        }
        leaderboard.append(row)
        log.info("%-20s CV %s = %.4f", name, cfg.scoring, search.best_score_)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    assert best_search is not None
    model = best_search.best_estimator_

    proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, proba, cfg.decision_threshold)
    ci_lo, ci_hi = bootstrap_ci(y_test, proba, seed=cfg.seed)
    test_metrics["roc_auc_ci95"] = [ci_lo, ci_hi]

    out = cfg.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    extras = save_plots(model, X_test, y_test, proba, out / "plots", cfg.decision_threshold, cfg.seed)

    metadata: dict[str, Any] = {
        "model_name": best_name,
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "seed": cfg.seed,
        "decision_threshold": cfg.decision_threshold,
        "selection_metric": cfg.scoring,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "data_fingerprint": data_fingerprint(X, y),
        "feature_names": list(X.columns),
        "class_names": {"0": "malignant", "1": "benign"},
        "best_params": {k: _py(v) for k, v in best_search.best_params_.items()},
        "leaderboard": sorted(
            leaderboard, key=lambda r: r[f"cv_{cfg.scoring}_mean"], reverse=True
        ),
        "test_metrics": test_metrics,
        "top_features": extras["top_features"],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    joblib.dump(model, out / MODEL_FILE)
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    log.info("Best model: %s | test ROC-AUC %.4f [%.4f, %.4f]",
             best_name, test_metrics["roc_auc"], ci_lo, ci_hi)
    log.info("Artifacts written to %s", Path(out).resolve())
    return metadata


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python so they are JSON serialisable."""
    return value.item() if isinstance(value, np.generic) else value
