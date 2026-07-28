"""Isolated tests for the inference business logic (without FastAPI).

We build tiny models right in the test, put them as artifacts in a temp directory,
spin up the registry + service and check both scenarios (regression/classification) and errors.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.repositories.model_registry import ModelRegistry
from app.services.inference import (
    InferenceService,
    InvalidFeaturesError,
    ModelNotFoundError,
)


def _make_registry(tmp_path: Path) -> ModelRegistry:
    """Create a registry with two trained toy models."""
    # Regression: y = 2*x.
    reg = LinearRegression().fit(pd.DataFrame({"x": [0, 1, 2, 3]}), [0, 2, 4, 6])
    joblib.dump(
        {
            "model": reg,
            "meta": {
                "name": "doubler",
                "task": "regression",
                "target": "y",
                "features": [{"name": "x", "type": "number"}],
            },
        },
        tmp_path / "doubler.joblib",
    )
    # Classification: class = (x > 0).
    clf = LogisticRegression().fit(pd.DataFrame({"x": [-2, -1, 1, 2]}), ["neg", "neg", "pos", "pos"])
    joblib.dump(
        {
            "model": clf,
            "meta": {
                "name": "sign",
                "task": "classification",
                "target": "cls",
                "features": [{"name": "x", "type": "number"}],
            },
        },
        tmp_path / "sign.joblib",
    )
    registry = ModelRegistry(tmp_path)
    registry.load()
    return registry


def test_regression_prediction(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    result = service.predict("doubler", {"x": 5})
    assert result.task == "regression"
    assert result.prediction == pytest.approx(10.0, abs=1e-6)


def test_classification_prediction(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    result = service.predict("sign", {"x": 3})
    assert result.task == "classification"
    assert result.prediction == "pos"
    assert result.probabilities is not None
    assert set(result.probabilities) == {"neg", "pos"}


def test_unknown_model_raises(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    with pytest.raises(ModelNotFoundError):
        service.predict("missing", {"x": 1})


def test_missing_feature_raises(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    with pytest.raises(InvalidFeaturesError):
        service.predict("doubler", {})
