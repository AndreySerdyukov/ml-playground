"""Train and export the REAL "Bayesian" model (wage) for ML Playground.

Source notebook is in `notebooks/bayesian/`, data is in `data/bayesian/` (in the repo itself).
NLS dataset: 545 young workers over 1980-1987 (~4360 "worker-year" rows). Goal - estimate
hourly wage from education, experience, union, marriage and race. In the notebook this is a Bayesian
linear regression (PyMC, StudentT); the final model `model_five` uses features
`Union, Expert, School, Married, Black`.

For prod we take its lightweight equivalent - sklearn `BayesianRidge` (a genuine Bayesian linear
regression: Gaussian priors on the weights, estimation via evidence). The artifact is tiny, PyMC is
not needed at runtime, and the coefficients reproduce the notebook's findings (union premium ~ +20%,
return on education ~ +10%/year). The `Wage` column is the LOGARITHM of wage; we train in log-space
and return `exp(Wage)` - dollars per hour (positive, intuitive), the same log-target trick as Cars.

Result - a bundle `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` in
`backend/models/bayesian.joblib`; it is read by the registry (`app/repositories/model_registry.py`),
the form is built from `features`. Alongside it - `bayesian.model_card.json` with metrics and provenance.
This file is the SINGLE source of truth for the description of the Bayesian model (`ModelInfo`).

Run (from the backend directory, in .venv):
    python training/bayesian.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Make the `app` package importable when running the file directly (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# The target column in the dataset is the LOGARITHM of hourly wage.
TARGET_COL = "Wage"

# Feature metadata: field type + human-readable label/unit/example.
FEATURE_META: dict[str, dict[str, Any]] = {
    "School": {"kind": "num", "label": "Years of schooling", "example": 12},
    "Expert": {"kind": "num", "label": "Work experience", "unit": "yrs", "example": 5},
    "Union": {"kind": "bin", "label": "Union member", "example": "Yes"},
    "Married": {"kind": "bin", "label": "Married", "example": "Yes"},
    "Black": {"kind": "bin", "label": "Black", "example": "No"},
}
# Order = order of form fields (as in the model_five notebook).
FEATURE_COLUMNS: list[str] = ["School", "Expert", "Union", "Married", "Black"]
# Binary columns: 0/1 in the data, shown as No/Yes in the form (dropdown).
BINARY_COLUMNS: list[str] = [n for n, m in FEATURE_META.items() if m["kind"] == "bin"]
BINARY_MAP: dict[float, str] = {0.0: "No", 1.0: "Yes"}

# Data lives in the repo itself: training/ -> backend/ -> repo-root(parents[2]) -> data/bayesian.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "bayesian" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "bayesian.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the dataset, drop rows without a target; binary 0/1 → No/Yes (as in the form)."""
    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=[TARGET_COL])
    for col in BINARY_COLUMNS:
        df[col] = df[col].astype(float).map(BINARY_MAP)
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec for the form: School/Expert - numbers, Union/Married/Black - a No/Yes dropdown."""
    specs: list[FeatureSpec] = []
    for name in FEATURE_COLUMNS:
        meta = FEATURE_META[name]
        if meta["kind"] == "num":
            specs.append(
                FeatureSpec(
                    name=name, type="number", label=meta["label"],
                    unit=meta.get("unit"), example=meta["example"],
                )
            )
        else:
            specs.append(
                FeatureSpec(
                    name=name, type="category", label=meta["label"],
                    choices=["No", "Yes"], example=meta["example"],
                )
            )
    return specs


def build_pipeline() -> TransformedTargetRegressor:
    """Preprocessing + BayesianRidge; the target ($/hr) is trained in log-space (= log-wage).

    Numeric (School/Expert) are passed through as is; binary ones are one-hot encoded (drop='if_binary' → 1 column).
    `TransformedTargetRegressor(log/exp)`: the model trains on log($/hr) = the original `Wage`, while predict
    returns exp(...) = $/hr. This exactly reproduces the notebook's log-regression but returns dollars.
    """
    num_features = [n for n in FEATURE_COLUMNS if FEATURE_META[n]["kind"] == "num"]
    preproc = ColumnTransformer(
        transformers=[
            # Imputer on top - so a missing value in the live form does not break inference (no NaN in the data).
            ("num", SimpleImputer(strategy="median"), num_features),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                BINARY_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    regressor = Pipeline(steps=[("preproc", preproc), ("estimator", BayesianRidge())])
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log, inverse_func=np.exp
    )


def evaluate(
    model: TransformedTargetRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Metrics on holdout ($/hr): mean MAPE, MEDIAN error, hit rate, MAE."""
    pred = np.clip(model.predict(x_test), 0.01, None)
    y = y_test.to_numpy(dtype=float)
    ape = np.abs((y - pred) / y)
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_15pct": float(np.mean(ape < 0.15)),
        "within_25pct": float(np.mean(ape < 0.25)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def build_model_info(specs: list[FeatureSpec], metrics: dict[str, float], n_train: int) -> ModelInfo:
    """Model card for the registry/form (is_stub=False - the "demo" badge disappears)."""
    median = metrics["median_ape"]
    within15 = metrics["within_15pct"]
    description = (
        f"Hourly-wage estimate – median error {median:.0%}, {within15:.0%} within 15% "
        f"(Bayesian linear regression on {n_train:,} worker-years)"
    )
    return ModelInfo(
        name="bayesian",
        task="regression",
        target="hourly wage",
        target_unit="$/hr",
        category="Regression",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Typical relative error (median APE) - the UI draws a range around the estimate.
        typical_error_pct=round(median, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Bayesian model (wage).")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Dataset not found: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    # `Wage` is log-wage; we keep the training/eval target in $/hr (exp), the pipeline does the log.
    y = np.exp(df[TARGET_COL].astype(float))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Training: {len(FEATURE_COLUMNS)} features, BayesianRidge, rows={len(df)} "
          f"(train={len(x_train)}, test={len(x_test)})")

    # Evaluation on holdout.
    eval_model = build_pipeline()
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(f"  median APE={metrics['median_ape']:.3f}  MAPE(mean)={metrics['mape']:.3f}  "
          f"within15%={metrics['within_15pct']:.0%}  MAE=${metrics['mae']:.2f}/hr")

    # Final model - on all the data.
    final_model = build_pipeline()
    final_model.fit(x, y)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_kb = args.out.stat().st_size / 1024
    print(f"  ✓ artifact: {args.out}  ({size_kb:.0f} KB)")

    # Model card (provenance + metrics) alongside the artifact; the registry reads only *.joblib.
    card = {
        "name": "bayesian",
        "task": "regression",
        "target": "exp(Wage) = $/hr",
        "estimator": "BayesianRidge + log-target",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Only the path tail - no absolute path with the username (we don't drag PII into the repo).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("bayesian.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
