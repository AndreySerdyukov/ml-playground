"""Isolated tests for the inference business logic (without FastAPI).

We build tiny models right in the test, put them as artifacts in a temp directory,
spin up the registry + service and check regression/classification/batch/uplift paths and errors.
A final smoke test loads the real shipped models if they have been built.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.config import get_settings
from app.repositories.model_registry import ModelRegistry
from app.services.file_ingest import parse_table
from app.services.inference import (
    InferenceService,
    InvalidFeaturesError,
    ModelNotFoundError,
    PredictionError,
)
from app.uplift import SoloModelUplift


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


# --- single predict ---

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


def test_bad_type_raises_prediction_error(tmp_path: Path) -> None:
    """A non-numeric value for a number feature is coerced to NaN; the toy model has no imputer,
    so the failure surfaces as a clean PredictionError (mapped to 422), never a raw 500."""
    service = InferenceService(_make_registry(tmp_path))
    with pytest.raises(PredictionError):
        service.predict("doubler", {"x": "not-a-number"})


# --- batch predict ---

def test_batch_prediction(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    resp = service.predict_batch("doubler", [{"x": 1}, {"x": 2}])
    assert resp.count == 2
    assert resp.columns == ["x"]
    assert resp.rows[0]["prediction"] == pytest.approx(2.0, abs=1e-6)
    assert resp.rows[1]["prediction"] == pytest.approx(4.0, abs=1e-6)


def test_batch_missing_column_raises(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    with pytest.raises(InvalidFeaturesError):
        service.predict_batch("doubler", [{"z": 1}])


def test_batch_empty_records_raises(tmp_path: Path) -> None:
    service = InferenceService(_make_registry(tmp_path))
    with pytest.raises(InvalidFeaturesError):
        service.predict_batch("doubler", [])


# --- uplift scenarios path ---

def test_uplift_scenarios(tmp_path: Path) -> None:
    """A SoloModelUplift artifact must produce a regression score plus the two treatment scenarios."""
    frame = pd.DataFrame({"f": [0, 1, 0, 1, 0, 1], "treatment_flg": [0, 0, 1, 1, 0, 1]})
    clf = LogisticRegression().fit(frame, [0, 0, 1, 1, 0, 1])
    wrap = SoloModelUplift(clf, ["f"], "treatment_flg")
    joblib.dump(
        {
            "model": wrap,
            "meta": {
                "name": "up",
                "task": "regression",
                "target": "uplift",
                "category": "Uplift",
                "features": [{"name": "f", "type": "number"}],
            },
        },
        tmp_path / "up.joblib",
    )
    registry = ModelRegistry(tmp_path)
    registry.load()
    result = InferenceService(registry).predict("up", {"f": 1})
    assert result.task == "regression"
    assert isinstance(result.prediction, float)
    assert result.scenarios is not None
    assert set(result.scenarios) == {"with_treatment", "without_treatment"}


# --- file ingest ---

def test_parse_table_csv_ok() -> None:
    records = parse_table("data.csv", b"x,y\n1,2\n3,4\n")
    assert records == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_parse_table_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_table("empty.csv", b"x,y\n")  # header only, no data rows


def test_parse_table_too_many_rows_raises() -> None:
    raw = b"n\n" + b"\n".join(str(i).encode() for i in range(5)) + b"\n"
    with pytest.raises(ValueError):
        parse_table("big.csv", raw, max_rows=3)


def test_parse_table_unreadable_raises() -> None:
    with pytest.raises(ValueError):
        parse_table("broken.xlsx", b"this is not a real xlsx file")


# --- smoke test on the real shipped models (skipped when artifacts are not built) ---

def test_real_models_smoke() -> None:
    registry = ModelRegistry(get_settings().models_dir)
    registry.load()
    infos = registry.list_infos()
    if not infos:
        pytest.skip("No built *.joblib artifacts in models/ (run the training scripts first)")
    service = InferenceService(registry)
    for info in infos:
        features = {
            f.name: (
                f.example
                if f.example is not None
                else (f.choices[0] if f.choices else 0)
            )
            for f in info.features
        }
        result = service.predict(info.name, features)
        assert result.model_name == info.name
        if result.task == "regression":
            assert isinstance(result.prediction, float)
            assert math.isfinite(result.prediction)
        else:
            assert result.probabilities is not None
            assert result.probabilities
