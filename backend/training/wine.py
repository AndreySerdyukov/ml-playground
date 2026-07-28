"""Train and export the REAL "Wine" (wine quality) model for ML Playground.

Source notebooks are in `notebooks/wine/`, data (train/test) in `data/wine/` (in the repo itself).
The task (as in the notebook) is BINARY: whether a white wine will be rated `quality >= 7` (class `Good`)
or lower (`Standard`). The classes are nearly balanced (~36% Good), so both precision and recall matter.

The notebook used logistic regression (linear, ROC-AUC ≈ 0.82). Here the model is **ExtraTrees**
(bagging of random trees): on this small noisy dataset the ensemble captures nonlinear
interactions (alcohol × density × sugar/SO₂) and lifts the entire precision/recall curve -
ROC-AUC ≈ 0.85, PR-AUC ≈ 0.74; at 90% precision recall grows from ~1.5% (logreg) to ~15%.
On top, `SimpleImputer(median)` - so a missing value in the live form does not break inference.

The result is a bundle `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` in
`backend/models/wine.joblib`; it is read by the registry (`app/repositories/model_registry.py`), and the form
is built from `features`. Alongside is `wine.model_card.json` with metrics and provenance.
This file is the SINGLE source of truth for the description of the Wine model (`ModelInfo`).

Run (from the backend directory, in .venv):
    python training/wine.py
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

# Make the `app` package importable when running the file directly (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo, ThresholdPoint

# Target column and the threshold for "good" wine.
TARGET_COL = "quality"
GOOD_THRESHOLD = 7
# Binary target -> human-readable class labels (the UI will show them).
GOOD_LABEL, BAD_LABEL = "Good", "Standard"
POSITIVE_LABEL = GOOD_LABEL

# Meta-description of the 11 physico-chemical features (all numeric). Examples are dataset medians
# (white wine), so "Try an example" fills a meaningful real row.
FEATURE_META: dict[str, dict[str, Any]] = {
    "fixed acidity": {"label": "Fixed acidity", "unit": "g/L", "example": 6.8},
    "volatile acidity": {"label": "Volatile acidity", "unit": "g/L", "example": 0.27},
    "citric acid": {"label": "Citric acid", "unit": "g/L", "example": 0.32},
    "residual sugar": {"label": "Residual sugar", "unit": "g/L", "example": 4.6},
    "chlorides": {"label": "Chlorides", "unit": "g/L", "example": 0.041},
    "free sulfur dioxide": {"label": "Free SO₂", "unit": "mg/L", "example": 33},
    "total sulfur dioxide": {"label": "Total SO₂", "unit": "mg/L", "example": 130},
    "density": {"label": "Density", "unit": "g/cm³", "example": 0.993},
    "pH": {"label": "pH", "unit": None, "example": 3.19},
    "sulphates": {"label": "Sulphates", "unit": "g/L", "example": 0.48},
    "alcohol": {"label": "Alcohol", "unit": "% vol", "example": 10.6},
}
FEATURE_COLUMNS: list[str] = list(FEATURE_META)

# Data lives in the repo itself: training/ -> backend/ -> repo-root(parents[2]) -> data/wine.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "wine" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "wine.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the dataset, drop duplicates and rows without a target."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def to_labels(quality: pd.Series) -> pd.Series:
    """Quality score -> string class Good/Standard (threshold >= GOOD_THRESHOLD)."""
    return pd.Series(
        np.where(quality.to_numpy() >= GOOD_THRESHOLD, GOOD_LABEL, BAD_LABEL), index=quality.index
    )


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec for the form: all 11 features are numeric (label/unit/example from FEATURE_META)."""
    return [
        FeatureSpec(
            name=name, type="number", label=meta["label"],
            unit=meta["unit"], example=meta["example"],
        )
        for name, meta in FEATURE_META.items()
    ]


def build_pipeline() -> Pipeline:
    """Imputer + ExtraTrees (tree bagging - stronger than a linear model on this dataset)."""
    return Pipeline(
        steps=[
            # Imputer on top - so a missing value in the live form does not break inference (no NaN in the data).
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_tuned() -> TunedThresholdClassifierCV:
    """Base pipeline + tuning the decision threshold for maximum F1 (class Good) via inner CV.

    A 0.5 threshold is too strict for a 36% minority class - it throttles recall. We tune by F1 (usually ~0.4),
    after which `predict` uses the selected threshold; `predict_proba`/`classes_` remain available to inference.
    """
    scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL)
    return TunedThresholdClassifierCV(
        build_pipeline(), scoring=scorer, cv=5, response_method="predict_proba", random_state=42
    )


def compute_metrics(
    y_bin: np.ndarray, base_proba_good: np.ndarray, tuned_pred: np.ndarray
) -> dict[str, Any]:
    """Metrics honestly on OOF: ranking ones - from base probabilities (threshold-independent), operating
    ones (precision/recall/F1/accuracy) - from the tuned classifier's predictions (nested CV, the threshold
    is picked on inner folds and never sees its own test rows).
    """
    tuned_bin = (tuned_pred == POSITIVE_LABEL).astype(int)  # 1 = Good
    return {
        # Model ranking (threshold-independent) - this is "how good the model is".
        "roc_auc": float(roc_auc_score(y_bin, base_proba_good)),
        "pr_auc": float(average_precision_score(y_bin, base_proba_good)),
        # Operating point at the selected threshold.
        "precision": float(precision_score(y_bin, tuned_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, tuned_bin)),
        "f1": float(f1_score(y_bin, tuned_bin)),
        "accuracy": float(accuracy_score(y_bin, tuned_bin)),
        # Confusion matrix in order [Standard(0), Good(1)].
        "confusion_matrix": confusion_matrix(y_bin, tuned_bin, labels=[0, 1]).tolist(),
        "labels": [BAD_LABEL, GOOD_LABEL],
    }


def threshold_curve(y_bin: np.ndarray, proba_good: np.ndarray) -> list[ThresholdPoint]:
    """Threshold -> (precision, recall) curve for class Good from OOF probabilities - for the UI slider."""
    pts: list[ThresholdPoint] = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        y_pred = (proba_good >= t).astype(int)
        pts.append(
            ThresholdPoint(
                threshold=float(t),
                precision=float(precision_score(y_bin, y_pred, zero_division=0)),
                recall=float(recall_score(y_bin, y_pred, zero_division=0)),
            )
        )
    return pts


def build_model_info(
    specs: list[FeatureSpec],
    metrics: dict[str, Any],
    n_rows: int,
    default_threshold: float,
    curve: list[ThresholdPoint],
) -> ModelInfo:
    """Model card for the registry/form (is_stub=False - the "demo" badge disappears)."""
    # Headline - threshold-independent ROC-AUC/PR-AUC (honest model "strength") + F1 (precision/recall summary
    # at the selected point). Accuracy and bare precision/recall do NOT go into the headline: the first is misleading
    # for an imbalanced task, the latter are meaningless without stating the threshold (see precision/recall in the model card).
    auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    f1 = metrics["f1"]
    description = (
        f"Good-wine (quality ≥ 7) classifier – ROC-AUC {auc:.2f}, PR-AUC {pr_auc:.2f}, F1 {f1:.2f} "
        f"(extra-trees, tuned threshold, on {n_rows:,} wines)"
    )
    return ModelInfo(
        name="wine",
        task="classification",
        target="wine quality",
        category="Classification",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Interactive threshold in the UI: slider starts at the selected threshold, positive class = Good.
        positive_class=POSITIVE_LABEL,
        default_threshold=round(default_threshold, 3),
        threshold_curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Wine (quality) model.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Dataset not found: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    y = to_labels(df[TARGET_COL])
    y_bin = (y == POSITIVE_LABEL).astype(int).to_numpy()
    good_share = float(y_bin.mean())
    print(f"Training: {len(FEATURE_COLUMNS)} features, ExtraTrees+tuned threshold, rows={len(df)}, "
          f"Good={good_share:.1%}")

    # Ranking metrics - from 5-fold OOF probabilities of the base model (threshold-independent, no leakage).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_proba = cross_val_predict(
        build_pipeline(), x, y_bin, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    # Operating point - NESTED CV: outer folds, and inside build_tuned picks the threshold on its own
    # folds, so precision/recall/F1 are honest (the threshold does not see its own test rows).
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tuned_pred = cross_val_predict(build_tuned(), x, y, cv=outer, n_jobs=-1)
    metrics = compute_metrics(y_bin, base_proba, tuned_pred)
    metrics["n_rows"] = len(df)
    metrics["eval"] = "5-fold OOF (ranking) + nested-CV tuned threshold (operating point)"
    print(f"  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  F1={metrics['f1']:.3f}  "
          f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"accuracy={metrics['accuracy']:.3f}")

    # Final model - on all data (string classes Good/Standard for the UI); threshold selected inside.
    final_model = build_tuned()
    final_model.fit(x, y)
    threshold = float(final_model.best_threshold_)
    metrics["threshold"] = threshold
    print(f"  selected threshold = {threshold:.3f}")

    # Threshold -> (precision, recall) curve from OOF - for the interactive slider in the UI.
    curve = threshold_curve(y_bin, base_proba)
    info = build_model_info(specs, metrics, len(df), threshold, curve)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ artifact: {args.out}  ({size_mb:.2f} MB)")

    # Model card (provenance + metrics) alongside the artifact; the registry reads only *.joblib.
    card = {
        "name": "wine",
        "task": "classification",
        "target": f"quality >= {GOOD_THRESHOLD} ({GOOD_LABEL} vs {BAD_LABEL})",
        "classes": [BAD_LABEL, GOOD_LABEL],
        "estimator": "ExtraTreesClassifier + TunedThresholdClassifierCV (F1)",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Only the path tail - without the absolute path containing the username (no PII in the repo).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("wine.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
