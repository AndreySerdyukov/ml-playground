"""Train and export the REAL "Diamonds" model for ML Playground.

Data (train/test/test_Y_true) is in `data/diamonds/` (inside the repo), the task metric is MAPE.

The MAIN LESSON of this script is the correct handling of outliers (in the source notebook it is broken
and inflated MAPE to 18.68 while the group best was ~7.31):
    - in the notebook the 1.5*IQR loop ran over ALL numeric columns, INCLUDING the target
      `total_sales_price`, and cut ~11% of rows out of training - the whole expensive tail (price > ~6700).
      The real `test.csv` keeps that tail → the model cannot reach expensive stones → MAPE blows up.
    - sometimes the same cutoff was applied BEFORE train_test_split → the test chunk got "cleaned" too (leakage,
      an optimistic 11.70 that turns into 18.68 on the real test).
Here is how it should be done: split FIRST, cleaning/imputation live INSIDE the sklearn pipeline (fit only on
train, transform on the rest → no leakage by construction), rows are NOT filtered by the TARGET, and the
external `test.csv`/`test_Y_true.csv` are not touched by hand until the final scoring.

Preprocessing:
    - numeric (size, depth/table %, meas_*): impossible zeros → NaN, then SimpleImputer(median)
      (analogous to IterativeImputer(missing_values=0) from the notebook, but honestly on train).
    - categorical (color, clarity, cut, symmetry, polish) - ordered grades →
      OrdinalEncoder with an explicit order (a monotonic grade code helps trees).
    - the target is learned in log space (log1p/expm1) OR via the native loss="gamma" - we take
      the best by internal validation (both variants optimize the RELATIVE error in line with MAPE).
Model - HistGradientBoostingRegressor (real weights, training in seconds).

Result - the bundle `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` in
`backend/models/diamonds.joblib`; it is read by the registry (`app/repositories/model_registry.py`),
the form is built from `features`. Alongside it - `diamonds.model_card.json` with metrics and provenance.
This file is the SINGLE source of truth for describing the Diamonds model (`ModelInfo`).

Run (from the backend directory, in .venv):
    python training/diamonds.py
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Make the `app` package importable when running the file directly (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Target column in the dataset (displayed as "price" in the UI).
TARGET_COL = "total_sales_price"

# Ordered grades (best → worst). One list per feature:
#   - goes into OrdinalEncoder as the explicit category order (monotonic code);
#   - and into the form's FeatureSpec.choices (choices == trained categories, unknown excluded).
GRADE_ORDER: dict[str, list[str]] = {
    "color": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
    "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"],
    "cut": ["Excellent", "Very Good"],
    "symmetry": ["Excellent", "Very Good"],
    "polish": ["Excellent", "Very Good"],
}

# Column metadata: field type + label/unit/example. Order = order of form fields.
FEATURE_META: dict[str, dict[str, Any]] = {
    "size": {"kind": "num", "label": "Carat", "unit": "ct", "example": 0.7},
    "color": {"kind": "cat", "label": "Colour", "example": "G"},
    "clarity": {"kind": "cat", "label": "Clarity", "example": "VS2"},
    "cut": {"kind": "cat", "label": "Cut", "example": "Excellent"},
    "symmetry": {"kind": "cat", "label": "Symmetry", "example": "Excellent"},
    "polish": {"kind": "cat", "label": "Polish", "example": "Excellent"},
    "depth_percent": {"kind": "num", "label": "Depth", "unit": "%", "example": 61.5},
    "table_percent": {"kind": "num", "label": "Table", "unit": "%", "example": 57.0},
    "meas_length": {"kind": "num", "label": "Length", "unit": "mm", "example": 5.7},
    "meas_width": {"kind": "num", "label": "Width", "unit": "mm", "example": 5.72},
    "meas_depth": {"kind": "num", "label": "Depth", "unit": "mm", "example": 3.53},
}
FEATURES: list[str] = list(FEATURE_META)
NUM_FEATURES = [c for c, m in FEATURE_META.items() if m["kind"] == "num"]
CAT_FEATURES = [c for c, m in FEATURE_META.items() if m["kind"] == "cat"]

# Data lives inside the repo: training/ -> backend/ -> repo-root(parents[2]) -> data/diamonds.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "diamonds"
DEFAULT_TRAIN_CSV = DATA_DIR / "train.csv"
DEFAULT_TEST_CSV = DATA_DIR / "test.csv"
DEFAULT_TEST_Y_CSV = DATA_DIR / "test_Y_true.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "diamonds.joblib"


def load_train(csv_path: Path) -> pd.DataFrame:
    """Read train, drop duplicates and priceless rows. Outliers are NOT trimmed by the TARGET."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec for the form; category choices = trained grades from GRADE_ORDER (in quality order)."""
    specs: list[FeatureSpec] = []
    for name in FEATURES:
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
                    choices=GRADE_ORDER[name], example=meta["example"],
                )
            )
    return specs


def build_pipeline(kind: str) -> Any:
    """Preprocessing + HGB. kind: 'log' (log target + squared_error) | 'gamma' (native loss)."""
    # Impossible zeros in numeric features (measurements/percents == 0) are fixed with the train median, then
    # a second imputer guards against an empty field in the live form (NaN). Both are built-in (picklable).
    numeric = Pipeline(
        steps=[
            ("zeros", SimpleImputer(missing_values=0, strategy="median")),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = OrdinalEncoder(
        categories=[GRADE_ORDER[c] for c in CAT_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_FEATURES),
            ("cat", categorical, CAT_FEATURES),
        ],
        remainder="drop",
    )

    common: dict[str, Any] = {
        "max_iter": 600,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 25,
        "random_state": 42,
    }
    if kind == "log":
        estimator = HistGradientBoostingRegressor(loss="squared_error", **common)
        regressor = Pipeline(steps=[("preproc", preproc), ("estimator", estimator)])
        # Log target: MSE in log space ≈ relative error (in line with MAPE) + price is always > 0.
        return TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1
        )
    if kind == "gamma":
        # Gamma loss models a positive skewed target with ~constant CV → also in line with MAPE.
        estimator = HistGradientBoostingRegressor(loss="gamma", **common)
        return Pipeline(steps=[("preproc", preproc), ("estimator", estimator)])
    raise ValueError(f"Unknown kind: {kind}")  # pragma: no cover


def evaluate(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Metrics: MAPE (mean), median error, hit rates, MAE, max error."""
    pred = np.clip(model.predict(x), 1, None)
    y_arr = y.to_numpy(dtype=float)
    ape = np.abs((y_arr - pred) / y_arr)
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape < 0.10)),
        "within_15pct": float(np.mean(ape < 0.15)),
        "within_25pct": float(np.mean(ape < 0.25)),
        "mae": float(mean_absolute_error(y_arr, pred)),
        "max_error": float(max_error(y_arr, pred)),
    }


def build_model_info(specs: list[FeatureSpec], test_metrics: dict[str, float], n_train: int) -> ModelInfo:
    """Model card for the registry/form (is_stub=False - the "demo" badge disappears)."""
    median = test_metrics["median_ape"]
    description = (
        f"Diamond price estimate – MAPE {test_metrics['mape']:.1%}, median error {median:.0%}, "
        f"{test_metrics['within_10pct']:.0%} within 10% (gradient boosting on {n_train:,} stones)"
    )
    return ModelInfo(
        name="diamonds",
        task="regression",
        target="price",
        target_unit="USD",
        category="Regression",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Typical relative error (median APE on the real test) - the UI draws a range around the estimate.
        typical_error_pct=round(median, 3),
    )


def _fmt(m: dict[str, float]) -> str:
    return (
        f"MAPE={m['mape']:.4f}  medianAPE={m['median_ape']:.4f}  "
        f"within10%={m['within_10pct']:.0%}  MAE={m['mae']:.0f} USD"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Diamonds model.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--test-y-csv", type=Path, default=DEFAULT_TEST_Y_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--kind", choices=["auto", "log", "gamma"], default="auto",
        help="model variant: auto = pick the best by internal validation",
    )
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Dataset not found: {args.train_csv}")

    df = load_train(args.train_csv)
    specs = build_feature_specs()
    x = df[FEATURES]
    y = df[TARGET_COL]

    # 1) Split FIRST. All cleaning/imputation is inside the pipeline (fit only on train). No leakage.
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"train.csv: rows={len(df)} (train={len(x_train)}, val={len(x_val)}); features={len(FEATURES)}")

    # 2) Choose the model variant by internal validation (or the one set explicitly).
    kinds = ["log", "gamma"] if args.kind == "auto" else [args.kind]
    val_scores: dict[str, dict[str, float]] = {}
    for kind in kinds:
        m = build_pipeline(kind)
        m.fit(x_train, y_train)
        val_scores[kind] = evaluate(m, x_val, y_val)
        print(f"  [{kind:5s}] holdout: {_fmt(val_scores[kind])}")
    best_kind = min(val_scores, key=lambda k: val_scores[k]["mape"])
    print(f"  → chosen variant: {best_kind}")

    # 3) Honest external scoring (comparable to 18.68 / the group's 7.31): train on the FULL train.csv,
    #    predict the external test.csv, compare against test_Y_true.csv. The test was not touched by hand.
    final_model = build_pipeline(best_kind)
    final_model.fit(x, y)
    test_metrics: dict[str, float] = {}
    if args.test_csv.exists() and args.test_y_csv.exists():
        x_test = pd.read_csv(args.test_csv)[FEATURES]
        y_test = pd.read_csv(args.test_y_csv)[TARGET_COL]
        test_metrics = evaluate(final_model, x_test, y_test)
        print(f"  EXTERNAL TEST (n={len(x_test)}): {_fmt(test_metrics)}  maxErr={test_metrics['max_error']:.0f}")
    else:
        print("  ⚠ test.csv/test_Y_true.csv not found - external scoring skipped, using holdout metrics")
        test_metrics = val_scores[best_kind]

    # 4) Export. We ship the model trained on train.csv (we do not learn on the test labels).
    info = build_model_info(specs, test_metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ artifact: {args.out}  ({size_mb:.2f} MB)")

    # Model card (provenance + metrics) alongside the artifact; the registry reads only *.joblib.
    card = {
        "name": "diamonds",
        "task": "regression",
        "target": TARGET_COL,
        "estimator": f"HistGradientBoostingRegressor ({best_kind})",
        "features": FEATURES,
        "metrics_holdout": val_scores.get(best_kind, {}),
        "metrics_external_test": test_metrics,
        "sklearn_version": sklearn.__version__,
        # Only the path tail - without the absolute path containing the username (no PII in the repo).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("diamonds.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
