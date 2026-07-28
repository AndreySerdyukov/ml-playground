"""Inference business logic. A pure layer: no FastAPI imports.

Responsible for: validating the feature set for a specific model, assembling the feature row
in the correct column order, and calling the estimator. Errors are domain exceptions
that the api layer translates into HTTP codes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.repositories.model_registry import ModelRegistry
from app.schemas.predict import BatchPredictResponse, ModelInfo, PredictResponse


class ModelNotFoundError(Exception):
    """A model was requested that is not in the registry."""


class InvalidFeaturesError(Exception):
    """The request is missing required model features."""


class InferenceService:
    """Orchestrates prediction on top of the model registry."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def list_models(self) -> list[ModelInfo]:
        """Descriptions of available models (for the UI and API)."""
        return self._registry.list_infos()

    def predict(self, model_name: str, features: dict[str, object]) -> PredictResponse:
        """Make a prediction with model `model_name` from a feature dict.

        The column order is taken from the model's metadata - the pipeline expects exactly that.
        """
        loaded = self._registry.get(model_name)
        if loaded is None:
            raise ModelNotFoundError(model_name)

        expected = [f.name for f in loaded.info.features]
        missing = [name for name in expected if name not in features]
        if missing:
            raise InvalidFeaturesError(f"Missing features: {', '.join(missing)}")

        row = pd.DataFrame([{name: features[name] for name in expected}])
        estimator = loaded.estimator

        if loaded.info.task == "classification":
            label = estimator.predict(row)[0]
            probabilities = self._extract_proba(estimator, row)
            return PredictResponse(
                model_name=model_name,
                task="classification",
                prediction=self._to_python(label),
                probabilities=probabilities,
            )

        value = float(estimator.predict(row)[0])
        # Uplift models (S-learner) additionally return outcome probabilities in the two treatment scenarios -
        # shown by the "rich" card. Other regressors do not have this method -> scenarios=None.
        scenarios = None
        if hasattr(estimator, "predict_scenarios"):
            p_treat, p_control = estimator.predict_scenarios(row)
            scenarios = {
                "with_treatment": float(p_treat[0]),
                "without_treatment": float(p_control[0]),
            }
        return PredictResponse(
            model_name=model_name, task="regression", prediction=value, scenarios=scenarios
        )

    def predict_batch(
        self, model_name: str, records: list[dict[str, object]]
    ) -> BatchPredictResponse:
        """Prediction over many rows at once (from an uploaded CSV/Excel).

        Returns feature values per row + a `prediction` column (row order is preserved).
        Numeric features are coerced to numbers; missing values and unknown categories are handled by the pipeline.
        """
        loaded = self._registry.get(model_name)
        if loaded is None:
            raise ModelNotFoundError(model_name)
        if not records:
            raise InvalidFeaturesError("No rows to predict")

        expected = [f.name for f in loaded.info.features]
        frame = pd.DataFrame(records)
        missing = [name for name in expected if name not in frame.columns]
        if missing:
            raise InvalidFeaturesError(f"The file is missing columns: {', '.join(missing)}")

        rows_in = frame[expected].copy()
        for feature in loaded.info.features:
            if feature.type == "number":
                rows_in[feature.name] = pd.to_numeric(rows_in[feature.name], errors="coerce")

        preds = loaded.estimator.predict(rows_in)
        is_cls = loaded.info.task == "classification"
        out_rows: list[dict[str, Any]] = []
        for i, record in enumerate(rows_in.to_dict("records")):
            row: dict[str, Any] = {name: self._cell(record[name]) for name in expected}
            # 4 digits: enough for monetary targets, and small uplift scores (~0.01-0.05) do not collapse.
            row["prediction"] = self._to_python(preds[i]) if is_cls else round(float(preds[i]), 4)
            out_rows.append(row)

        return BatchPredictResponse(
            model_name=model_name,
            task=loaded.info.task,
            target=loaded.info.target,
            target_unit=loaded.info.target_unit,
            columns=expected,
            rows=out_rows,
            count=len(out_rows),
        )

    @staticmethod
    def _cell(value: object) -> object:
        """Cell scalar -> a JSON-compatible type (NaN/missing -> None)."""
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _extract_proba(estimator: object, row: pd.DataFrame) -> dict[str, float] | None:
        """Return per-class probabilities if the estimator supports them."""
        if not hasattr(estimator, "predict_proba"):
            return None
        proba = estimator.predict_proba(row)[0]
        classes = [str(c) for c in getattr(estimator, "classes_", range(len(proba)))]
        return {cls: float(p) for cls, p in zip(classes, proba)}

    @staticmethod
    def _to_python(value: object) -> float | int | str:
        """Coerce a numpy scalar to a native Python type for serialization."""
        if isinstance(value, np.generic):
            return value.item()
        return value  # type: ignore[return-value]
