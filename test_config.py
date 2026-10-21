from pathlib import Path

import pytest

from diagml.config import Config, ModelSpec, load_config


def test_default_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.cv_folds >= 2
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(cfg.models)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        Config(test_size=1.5, models={"m": ModelSpec()}).validate()
    with pytest.raises(ValueError):
        Config(cv_folds=1, models={"m": ModelSpec()}).validate()
    with pytest.raises(ValueError):
        Config().validate()  # no models
