"""Input/output schemas for inference and model descriptions."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Model task type (uplift is modeled as regression - a numeric score).
TaskType = Literal["regression", "classification"]


class FeatureSpec(BaseModel):
    """Description of a single input feature (for auto-generating the form on the frontend)."""

    name: str
    # Field type for the UI: number or category (dropdown list).
    type: Literal["number", "category"]
    # Allowed values for a categorical feature.
    choices: list[str] | None = None
    # Human-readable field label.
    label: str | None = None
    # Unit of measurement (for the label), e.g. "USD", "km".
    unit: str | None = None
    # Default value for the "Try an example" button.
    example: float | str | None = None


class ThresholdPoint(BaseModel):
    """A threshold-curve point: precision/recall of the positive class at the given threshold (from OOF)."""

    threshold: float
    precision: float
    recall: float


class ModelInfo(BaseModel):
    """Public description of a model in the registry (catalog card + form)."""

    name: str
    task: TaskType
    target: str
    description: str = ""
    features: list[FeatureSpec] = Field(default_factory=list)
    # Emoji icon for the catalog card.
    emoji: str = ""
    # Displayed category: "Regression" | "Classification" | "Uplift".
    category: str = ""
    # Unit of measurement of the target (for the result card).
    target_unit: str | None = None
    # Typical relative regression error (median APE, a fraction) - the UI draws a range around the estimate.
    typical_error_pct: float | None = None
    # Whether this is a stub (weights not plugged in yet) - the UI shows a "demo" badge.
    is_stub: bool = False
    # --- BINARY classification only: interactive decision threshold in the UI ---
    # The class to whose probability the threshold is applied (e.g. "Good"). None -> no slider.
    positive_class: str | None = None
    # The model's working threshold (starting slider position); None -> 0.5.
    default_threshold: float | None = None
    # Threshold->(precision, recall) curve from OOF - the UI shows metrics at the chosen threshold.
    threshold_curve: list[ThresholdPoint] | None = None


class PredictRequest(BaseModel):
    """Prediction request: model name + feature values."""

    features: dict[str, Any]


class PredictResponse(BaseModel):
    """Inference response."""

    model_name: str
    task: TaskType
    # Prediction: a number (regression) or a class label (classification).
    prediction: float | str | int
    # For classification - per-class probabilities (if the model returns them).
    probabilities: dict[str, float] | None = None
    # For uplift - outcome probabilities in the two treatment scenarios (with_treatment / without_treatment);
    # the "rich" card draws both scales and a verdict. None for other models.
    scenarios: dict[str, float] | None = None


class BatchPredictResponse(BaseModel):
    """Batch predict response: feature columns + rows with values and a prediction."""

    model_name: str
    task: TaskType
    target: str
    target_unit: str | None = None
    # Feature column names in order (the header of the results table on the frontend).
    columns: list[str]
    # Per row: feature values + a "prediction" key.
    rows: list[dict[str, Any]]
    count: int
