"""Train and export the REAL "Loans" (credit approval) model for ML Playground.

Source notebooks are in `notebooks/loans/`, data (train/test) in `data/loans/` (in the repo itself).
The task is binary classification: approve (`Approved`) or reject (`Rejected`) an application.
The data is ANONYMIZED: columns are nameless (`C1, N2, C4_enc, …`), meaning hidden for
confidentiality. All features are fed to the model as numbers (`*_enc` are arbitrary categorical
codes, but tree ensembles tolerate this) - exactly as in the original notebook.

The model reproduces the final notebook `ML_Classification_Credits_Model.ipynb` -
a `StackingClassifier` of three base learners (GradientBoosting + AdaBoost + RandomForest) with
a LogisticRegression meta-learner (`stack_method="predict_proba"`). On top - `SimpleImputer(median)`
in case of missing values in the live form. The 1/0 target is mapped to string classes `Approved`/`Rejected`,
so the UI labels the probability bars human-readably.

The result is a bundle `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` in
`backend/models/loans.joblib`; it is read by the registry (`app/repositories/model_registry.py`),
and the form is built from `features`. Alongside is `loans.model_card.json` with metrics and provenance.
This file is the SINGLE source of truth for the description of the Loans model (`ModelInfo`).

Run (from the backend directory, in .venv):
    python training/loans.py
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Make the `app` package importable when running the file directly (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo, ThresholdPoint

# Target column in the dataset.
TARGET_COL = "Target"
# 1/0 target -> human-readable class labels (the UI will show them).
CLASS_LABELS: dict[float, str] = {1.0: "Approved", 0.0: "Rejected"}
# Positive class for metrics (precision/recall/ROC-AUC).
POSITIVE_LABEL = "Approved"

# Features in dataset column order (= form field order). All numeric.
# Names are ANONYMOUS - shown as is (without invented labels).
FEATURE_COLUMNS: list[str] = [
    "C1", "N2", "N3", "C4_enc", "C5_enc", "C6_enc", "N7",
    "C8", "C9", "N10", "C11", "C12_enc", "N13", "N14",
]

# Values of one real (approved) train row - for the "Try an example" button.
# Keys are kept in sync with FEATURE_COLUMNS (build_feature_specs reads by each name).
FEATURE_EXAMPLES: dict[str, float] = {
    "C1": 1.0, "N2": 28.0, "N3": 2.0, "C4_enc": 2.0, "C5_enc": 4.0,
    "C6_enc": 8.0, "N7": 4.165, "C8": 1.0, "C9": 1.0, "N10": 2.0,
    "C11": 1.0, "C12_enc": 2.0, "N13": 181.0, "N14": 1.0,
}

# Data lives in the repo itself: training/ -> backend/ -> repo-root(parents[2]) -> data/loans.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "loans" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "loans.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the dataset, drop duplicates and rows without a target."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec for the form: all 14 features are numeric, label = raw anonymous name."""
    return [
        FeatureSpec(
            name=name, type="number", label=name, example=FEATURE_EXAMPLES[name]
        )
        for name in FEATURE_COLUMNS
    ]


def build_pipeline() -> Pipeline:
    """Imputer + stacking ensemble (reproduces the final notebook).

    Base learners: GradientBoosting / AdaBoost(stumps) / RandomForest; meta: StandardScaler+LogReg.
    Hyperparameters are from the notebook; `random_state=42` added for a deterministic artifact.
    """
    grad = GradientBoostingClassifier(
        learning_rate=0.11,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=4,
        n_estimators=100,
        subsample=0.9,
        random_state=42,
    )
    forest = RandomForestClassifier(
        criterion="gini",
        max_depth=4,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
    )
    # sklearn 1.9: the parameter is named `estimator` (`base_estimator` deprecated in 1.2, removed in 1.4).
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    )
    # L1 regularization: in sklearn 1.9 `penalty="l1"` -> `l1_ratio=1` (gives the same coefficients).
    final_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, l1_ratio=1, solver="liblinear", random_state=42),
    )
    stacking = StackingClassifier(
        estimators=[("grad", grad), ("ada", ada), ("forest", forest)],
        final_estimator=final_estimator,
        stack_method="predict_proba",
    )
    # Imputer on top - so a missing value in the live form does not break inference (training data is complete).
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", stacking),
        ]
    )


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Holdout metrics: accuracy, ROC-AUC, precision/recall/F1 (class Approved), confusion matrix."""
    pred = model.predict(x_test)
    classes = list(model.classes_)
    pos_idx = classes.index(POSITIVE_LABEL)
    proba_pos = model.predict_proba(x_test)[:, pos_idx]
    y_true_bin = (y_test.to_numpy() == POSITIVE_LABEL).astype(int)
    # Label order for the confusion matrix - the single source of truth CLASS_LABELS.
    labels = [CLASS_LABELS[0.0], CLASS_LABELS[1.0]]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_true_bin, proba_pos)),
        "precision": float(precision_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        "recall": float(recall_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        "f1": float(f1_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        # Confusion matrix in order labels=[Rejected, Approved].
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "labels": labels,
    }


def threshold_curve(y_bin: np.ndarray, proba_pos: np.ndarray) -> list[ThresholdPoint]:
    """Threshold -> (precision, recall) curve for class Approved from OOF probabilities - for the UI slider."""
    pts: list[ThresholdPoint] = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        y_pred = (proba_pos >= t).astype(int)
        pts.append(
            ThresholdPoint(
                threshold=float(t),
                precision=float(precision_score(y_bin, y_pred, zero_division=0)),
                recall=float(recall_score(y_bin, y_pred, zero_division=0)),
            )
        )
    return pts


def build_model_info(
    specs: list[FeatureSpec], metrics: dict[str, Any], n_rows: int, curve: list[ThresholdPoint]
) -> ModelInfo:
    """Model card for the registry/form (is_stub=False - the "demo" badge disappears)."""
    auc = metrics["roc_auc"]
    acc = metrics["accuracy"]
    # Metrics - from holdout; n_rows - the size of the dataset the final model is trained on.
    description = (
        f"Credit approval classifier – ROC-AUC {auc:.2f}, accuracy {acc:.0%} "
        f"(stacked ensemble on {n_rows:,} anonymised applications)"
    )
    return ModelInfo(
        name="loans",
        task="classification",
        target="credit decision",
        category="Classification",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Interactive threshold in the UI: positive class = Approved, slider starts at the argmax threshold 0.5.
        positive_class=POSITIVE_LABEL,
        default_threshold=0.5,
        threshold_curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Loans model.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Dataset not found: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COL].astype(float).map(CLASS_LABELS)
    if y.isna().any():
        raise SystemExit("The target has values outside {0, 1} - check the dataset.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Training: {len(FEATURE_COLUMNS)} features, stacking ensemble, "
        f"rows={len(df)} (train={len(x_train)}, test={len(x_test)})"
    )

    # Holdout evaluation.
    eval_model = build_pipeline()
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(
        f"  accuracy={metrics['accuracy']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  "
        f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}"
    )
    print(f"  confusion[Rejected/Approved]={metrics['confusion_matrix']}")

    # Final model - on all data.
    final_model = build_pipeline()
    final_model.fit(x, y)

    # Threshold -> (precision, recall) curve from 5-fold OOF - for the interactive slider in the UI.
    y_bin = (y == POSITIVE_LABEL).astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_proba = cross_val_predict(
        build_pipeline(), x, y_bin, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    curve = threshold_curve(y_bin, base_proba)

    info = build_model_info(specs, metrics, len(df), curve)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ artifact: {args.out}  ({size_mb:.2f} MB)")

    # Model card (provenance + metrics) alongside the artifact; the registry reads only *.joblib.
    card = {
        "name": "loans",
        "task": "classification",
        "target": TARGET_COL,
        "classes": [CLASS_LABELS[0.0], CLASS_LABELS[1.0]],
        "estimator": "stacking(grad+ada+forest -> logreg)",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Only the path tail - without the absolute path containing the username (no PII in the repo).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("loans.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
