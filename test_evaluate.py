import numpy as np

from diagml.evaluate import bootstrap_ci, compute_metrics


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, proba)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_threshold_changes_predictions():
    y = np.array([0, 1, 1, 1])
    proba = np.array([0.4, 0.6, 0.7, 0.45])
    assert compute_metrics(y, proba, 0.5)["fn"] == 1
    assert compute_metrics(y, proba, 0.3)["fn"] == 0


def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    lo, hi = bootstrap_ci(y, proba, n_boot=100)
    assert 0.5 < lo <= hi <= 1.0
