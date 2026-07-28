"""Train and export the REAL "Cars" model for ML Playground.

Source notebooks are in `notebooks/cars/`, data (train/test) is in `data/cars/` (inside the repo).
Preprocessing follows the spirit of the source notebook:
    - numeric: SimpleImputer(median)
    - categorical: SimpleImputer(constant "Unknown") + OneHotEncoder(handle_unknown="ignore")
    - the priceUSD target is learned in log space (log1p/expm1 via TransformedTargetRegressor):
      this optimizes the RELATIVE error (in line with MAPE) and guarantees a positive price.
Model - a compact HistGradientBoostingRegressor: real weights, artifact <1 MB, training in seconds.

Demo domain - ORDINARY used cars: `for parts`/`with damage` rows and absurd price/mileage are dropped
(they produce a wild relative error and are not interesting as a "car valuation"). Hence `condition`
is removed from the features (within the domain it is a constant). The card metric is the MEDIAN error
and the hit rate (mean MAPE inflates the cheap tail; the MAPE ceiling of these features is ~14%, see progress).

Result - the bundle `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` in
`backend/models/cars.joblib`; it is read by the registry (`app/repositories/model_registry.py`),
the form is built from `features`. Alongside it - `cars.model_card.json` with metrics and provenance.
This file is the SINGLE source of truth for describing the Cars model (`ModelInfo`).

Run (from the backend directory, in .venv):
    python training/cars.py                 # prod set, clean domain (default)
    python training/cars.py --feature-set A --full   # all 11 features, no domain filter
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Make the `app` package importable when running the file directly (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Target column in the dataset (displayed as "price" in the UI).
TARGET_COL = "priceUSD"
# Value the imputer uses to fill missing categories (a real trained category).
UNKNOWN_FILL = "Unknown"

# Demo domain: ordinary running used cars, without scrap/damaged and absurd values.
DOMAIN_CONDITION = "with mileage"
PRICE_MIN, PRICE_MAX = 500, 120_000
MILEAGE_MAX = 700_000

# Metadata for all potential columns: field type + human-readable label/unit/example.
# `choices` for categories are NOT hardcoded - we take them from the data (guarantees a match with the trained OHE).
FEATURE_META: dict[str, dict[str, Any]] = {
    "company": {"kind": "cat", "label": "Make", "example": "volkswagen"},
    "model": {"kind": "cat", "label": "Model", "example": "passat"},
    "year": {"kind": "num", "label": "Year", "example": 2012},
    "mileage(km)": {"kind": "num", "label": "Mileage", "unit": "km", "example": 180000},
    "volume(cm3)": {"kind": "num", "label": "Engine volume", "unit": "cm³", "example": 1800},
    "fuel": {"kind": "cat", "label": "Fuel", "example": "petrol"},
    "transmission": {"kind": "cat", "label": "Transmission", "example": "auto"},
    "drive_unit": {"kind": "cat", "label": "Drive", "example": "front-wheel drive"},
    "condition": {"kind": "cat", "label": "Condition", "example": "with mileage"},
    "color": {"kind": "cat", "label": "Colour", "example": "black"},
    "vehicle_size_class": {"kind": "cat", "label": "Body class", "example": "D"},
}

# Feature sets. Order = order of form fields.
FEATURE_SETS: dict[str, list[str]] = {
    # prod (default) - 10 features without condition (domain filtered to "with mileage").
    "prod": [
        "company", "model", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "color", "vehicle_size_class",
    ],
    # A - full reproduction: all 11 features (experiment, usually with --full).
    "A": [
        "company", "model", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "condition", "color", "vehicle_size_class",
    ],
    # B - curated: without the form-killers model/color.
    "B": [
        "company", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "vehicle_size_class",
    ],
    # C - as in the original stub: exactly 6 fields.
    "C": ["company", "year", "mileage(km)", "fuel", "volume(cm3)", "transmission"],
}

# Data lives inside the repo: training/ -> backend/ -> repo-root(parents[2]) -> data/cars.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "cars" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "cars.joblib"


def load_data(csv_path: Path, clean_domain: bool = True) -> pd.DataFrame:
    """Read the dataset, drop duplicates/priceless rows; optionally narrow to the demo domain of ordinary cars."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    # Cut off zero/broken prices: they both spoil the model and make MAPE infinite.
    df = df[df[TARGET_COL] > 0]
    if clean_domain:
        df = df[
            (df["condition"] == DOMAIN_CONDITION)
            & df[TARGET_COL].between(PRICE_MIN, PRICE_MAX)
            & df["mileage(km)"].between(1, MILEAGE_MAX)
        ]
    return df.reset_index(drop=True)


def build_feature_specs(df: pd.DataFrame, columns: list[str]) -> list[FeatureSpec]:
    """Build FeatureSpec for the form; category choices are taken from the unique values in the data."""
    specs: list[FeatureSpec] = []
    for name in columns:
        meta = FEATURE_META[name]
        if meta["kind"] == "num":
            specs.append(
                FeatureSpec(
                    name=name, type="number", label=meta["label"],
                    unit=meta.get("unit"), example=meta["example"],
                )
            )
        else:
            choices = sorted(str(v) for v in df[name].dropna().unique())
            # If the column had missing values - the imputer trained the UNKNOWN_FILL category,
            # expose it as a separate valid option ("don't know").
            if bool(df[name].isna().any()):
                choices.append(UNKNOWN_FILL)
            specs.append(
                FeatureSpec(
                    name=name, type="category", label=meta["label"],
                    choices=choices, example=meta["example"],
                )
            )
    return specs


def build_pipeline(
    num_features: list[str], cat_features: list[str], estimator_name: str
) -> TransformedTargetRegressor:
    """Preprocessing + regressor; the target is learned in log space (relative error)."""
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=UNKNOWN_FILL)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric, num_features),
            ("cat", categorical, cat_features),
        ],
        remainder="drop",
    )

    if estimator_name == "hgb":
        estimator: Any = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=42,
        )
    elif estimator_name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=300, min_samples_split=5, n_jobs=-1, random_state=42
        )
    else:  # pragma: no cover - guard against a typo in the argument
        raise ValueError(f"Unknown estimator: {estimator_name}")

    regressor = Pipeline(steps=[("preproc", preproc), ("estimator", estimator)])
    # Log target: MSE in log space ≈ relative error (in line with MAPE) + price is always > 0.
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )


def evaluate(
    model: TransformedTargetRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Holdout metrics: mean MAPE, MEDIAN error, hit rate, MAE."""
    pred = np.clip(model.predict(x_test), 1, None)
    y = y_test.to_numpy(dtype=float)
    ape = np.abs((y - pred) / y)
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape < 0.10)),
        "within_15pct": float(np.mean(ape < 0.15)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def build_model_info(specs: list[FeatureSpec], metrics: dict[str, float], n_train: int) -> ModelInfo:
    """Model card for the registry/form (is_stub=False - the "demo" badge disappears)."""
    median = metrics["median_ape"]
    within10 = metrics["within_10pct"]
    description = (
        f"Used-car price estimate – MAPE {metrics['mape']:.1%}, median error {median:.0%}, "
        f"{within10:.0%} within 10% (gradient boosting on {n_train:,} listings)"
    )
    return ModelInfo(
        name="cars",
        task="regression",
        target="price",
        target_unit="USD",
        category="Regression",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Typical relative error (median APE) - the UI draws a range around the estimate.
        typical_error_pct=round(median, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Cars model.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--estimator", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="prod")
    parser.add_argument(
        "--full", action="store_true", help="do not narrow the domain (whole dataset, including scrap/damaged)"
    )
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Dataset not found: {args.train_csv}")

    clean_domain = not args.full
    df = load_data(args.train_csv, clean_domain=clean_domain)
    columns = FEATURE_SETS[args.feature_set]
    specs = build_feature_specs(df, columns)
    num_features = [s.name for s in specs if s.type == "number"]
    cat_features = [s.name for s in specs if s.type == "category"]

    x = df[columns]
    y = df[TARGET_COL]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    domain = "ordinary cars" if clean_domain else "whole dataset"
    print(
        f"Training: feature-set={args.feature_set} ({len(columns)} features), estimator={args.estimator}, "
        f"domain={domain}, rows={len(df)} (train={len(x_train)}, test={len(x_test)})"
    )

    # Evaluation on holdout.
    eval_model = build_pipeline(num_features, cat_features, args.estimator)
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(
        f"  MAPE(mean)={metrics['mape']:.3f}  median APE={metrics['median_ape']:.3f}  "
        f"within10%={metrics['within_10pct']:.0%}  MAE={metrics['mae']:.0f} USD"
    )

    # Final model - on all data.
    final_model = build_pipeline(num_features, cat_features, args.estimator)
    final_model.fit(x, y)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ artifact: {args.out}  ({size_mb:.2f} MB)")

    # Model card (provenance + metrics) alongside the artifact; the registry reads only *.joblib.
    card = {
        "name": "cars",
        "task": "regression",
        "target": TARGET_COL,
        "estimator": args.estimator,
        "feature_set": args.feature_set,
        "features": columns,
        "clean_domain": clean_domain,
        # Metric scope: with clean_domain the reported MAPE covers ONLY ordinary running cars
        # (condition="with mileage", price 500-120000 USD, mileage 1-700000 km). Scrap/damaged and
        # absurd listings are excluded by design - the headline error is for this domain, not the raw feed.
        "metric_scope": (
            "ordinary running cars only (condition='with mileage', price 500-120000 USD, "
            "mileage 1-700000 km); extreme/scrap listings excluded by design"
            if clean_domain else "whole dataset (no domain filter)"
        ),
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Only the path tail - without the absolute path containing the username (no PII in the repo).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("cars.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
