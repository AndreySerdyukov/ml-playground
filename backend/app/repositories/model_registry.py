"""Model registry: loads `*.joblib` artifacts from the `models/` directory.

Artifact format - dict:
    {"model": <fitted sklearn estimator/pipeline>, "meta": {...ModelInfo...}}

This is the application's data layer: the only place that knows where models come from.
The inference business logic (`services/`) only uses the registry but knows nothing about the files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.schemas.predict import ModelInfo


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model: the estimator itself + its metadata description."""

    estimator: Any
    info: ModelInfo


class ModelRegistry:
    """Keeps models in memory, reading them lazily from disk on initialization."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models: dict[str, LoadedModel] = {}

    def load(self) -> None:
        """Read all `*.joblib` from the models directory into memory."""
        self._models.clear()
        if not self._models_dir.exists():
            return
        for path in sorted(self._models_dir.glob("*.joblib")):
            bundle = joblib.load(path)
            info = ModelInfo.model_validate(bundle["meta"])
            self._models[info.name] = LoadedModel(estimator=bundle["model"], info=info)

    def list_infos(self) -> list[ModelInfo]:
        """List of descriptions of all available models."""
        return [m.info for m in self._models.values()]

    def get(self, name: str) -> LoadedModel | None:
        """Model by name, or None if not found."""
        return self._models.get(name)
