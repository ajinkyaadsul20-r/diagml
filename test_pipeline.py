import json

import pytest

from diagml.data import load_data
from diagml.models import build_pipeline
from diagml.predict import Predictor


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_pipeline("nope", 0)


def test_training_writes_artifacts(trained_dir):
    for name in ("model.joblib", "metadata.json"):
        assert (trained_dir / name).exists()
    for plot in ("roc_curve", "confusion_matrix", "calibration", "feature_importance"):
        assert (trained_dir / "plots" / f"{plot}.png").exists()
    meta = json.loads((trained_dir / "metadata.json").read_text())
    assert meta["test_metrics"]["roc_auc"] > 0.95
    assert len(meta["leaderboard"]) == 2
    assert len(meta["feature_names"]) == 30


def test_predictor_end_to_end(trained_dir):
    X, y = load_data()
    predictor = Predictor(trained_dir)
    out = predictor.predict_one(X.iloc[0].to_dict())
    assert out["class_name"] in {"malignant", "benign"}
    assert abs(out["probability_benign"] + out["probability_malignant"] - 1) < 1e-4
    assert out["class_name"] == "malignant"  # row 0 is malignant in this dataset


def test_predictor_batch_accuracy(trained_dir):
    X, y = load_data()
    predictor = Predictor(trained_dir)
    preds = predictor.predict_many([X.iloc[i].to_dict() for i in range(50)])
    acc = sum(p["label"] == y.iloc[i] for i, p in enumerate(preds)) / 50
    assert acc > 0.9


def test_predictor_rejects_bad_input(trained_dir):
    predictor = Predictor(trained_dir)
    with pytest.raises(ValueError, match="Missing"):
        predictor.predict_one({"mean radius": 1.0})
    X, _ = load_data()
    rec = X.iloc[0].to_dict()
    rec["bogus"] = 1.0
    with pytest.raises(ValueError, match="Unknown"):
        predictor.predict_one(rec)


def test_predictor_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)
