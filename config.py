"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """Search space for one candidate model."""

    params: dict[str, list[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "roc_auc"
    n_iter: int = 12
    n_jobs: int = -1
    decision_threshold: float = 0.5
    artifacts_dir: Path = Path("artifacts")
    models: dict[str, ModelSpec] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1)")
        if not self.models:
            raise ValueError("config must define at least one model")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config. Falls back to built-in defaults when no file exists."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    models = {
        name: ModelSpec(params=(spec or {}).get("params", {}))
        for name, spec in raw.pop("models", {}).items()
    }
    if "artifacts_dir" in raw:
        raw["artifacts_dir"] = Path(raw["artifacts_dir"])
    cfg = Config(models=models, **raw)
    cfg.validate()
    return cfg
