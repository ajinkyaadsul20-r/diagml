"""Inference wrapper shared by the CLI and the REST API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .train import METADATA_FILE, MODEL_FILE


class Predictor:
    """Loads a trained model plus its metadata and validates inputs before predicting."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        art = Path(artifacts_dir)
        model_path, meta_path = art / MODEL_FILE, art / METADATA_FILE
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No trained model in '{art}'. Run `diagml train` first."
            )
        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text())
        self.feature_names: list[str] = self.metadata["feature_names"]
        self.threshold: float = float(self.metadata["decision_threshold"])
        self.class_names: dict[int, str] = {
            int(k): v for k, v in self.metadata["class_names"].items()
        }

    def validate(self, record: dict[str, float]) -> pd.DataFrame:
        missing = [f for f in self.feature_names if f not in record]
        unknown = [k for k in record if k not in self.feature_names]
        if missing or unknown:
            raise ValueError(f"Invalid input. Missing: {missing}. Unknown: {unknown}.")
        try:
            row = {f: float(record[f]) for f in self.feature_names}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"All features must be numeric: {exc}") from exc
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_one(self, record: dict[str, float]) -> dict[str, Any]:
        return self.predict_many([record])[0]

    def predict_many(self, records: list[dict[str, float]]) -> list[dict[str, Any]]:
        frame = pd.concat([self.validate(r) for r in records], ignore_index=True)
        proba = self.model.predict_proba(frame)[:, 1]
        results = []
        for p in proba:
            label = int(p >= self.threshold)
            results.append(
                {
                    "label": label,
                    "class_name": self.class_names[label],
                    "probability_benign": round(float(p), 6),
                    "probability_malignant": round(float(1 - p), 6),
                }
            )
        return results
