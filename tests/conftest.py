"""Shared fixtures. A tiny, fast config keeps the test-suite quick."""

from __future__ import annotations

import pytest

from diagml.config import Config, ModelSpec
from diagml.train import train


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = Config(
        seed=0,
        cv_folds=3,
        n_iter=2,
        n_jobs=1,
        artifacts_dir=out,
        models={
            "logistic_regression": ModelSpec(params={"clf__C": [0.1, 1.0]}),
            "random_forest": ModelSpec(params={"clf__n_estimators": [20, 40]}),
        },
    )
    train(cfg)
    return out
