"""FastAPI service. Run with: uvicorn diagml.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .predict import Predictor

ARTIFACTS_DIR = os.getenv("DIAGML_ARTIFACTS_DIR", "artifacts")
_state: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["predictor"] = Predictor(ARTIFACTS_DIR)
    yield
    _state.clear()


app = FastAPI(
    title="diagml API",
    version=__version__,
    description="Predict tumour class (benign / malignant) from 30 cell-nucleus features. "
    "Educational project, not a medical device.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Mapping of all 30 feature names to numeric values."
    )


class BatchRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., min_length=1, max_length=1000)


def _predictor() -> Predictor:
    if "predictor" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _state["predictor"]


@app.get("/health")
def health() -> dict[str, str]:
    _predictor()
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info() -> dict[str, Any]:
    meta = _predictor().metadata
    return {
        "model_name": meta["model_name"],
        "trained_at": meta["trained_at"],
        "data_fingerprint": meta["data_fingerprint"],
        "feature_names": meta["feature_names"],
        "test_metrics": meta["test_metrics"],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return _predictor().predict_one(req.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict[str, Any]:
    try:
        return {"predictions": _predictor().predict_many(req.records)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
