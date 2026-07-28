"""Обучение и экспорт РЕАЛЬНОЙ модели «Wine» (качество вина) для ML Playground.

Исходные ноутбуки – в `notebooks/wine/`, данные (train/test) – в `data/wine/` (в самом репо).
Задача (как в ноутбуке) – БИНАРНАЯ: будет ли белое вино оценено на `quality >= 7` (класс `Good`)
или ниже (`Standard`). Классы почти сбалансированы (~36% Good), поэтому важны и precision, и recall.

Ноутбук брал логистическую регрессию (линейная, ROC-AUC ≈ 0.82). Здесь модель – **ExtraTrees**
(бэггинг случайных деревьев): на этом небольшом шумном датасете ансамбль ловит нелинейные
взаимодействия (alcohol × density × сахар/SO₂) и поднимает всю precision/recall-кривую –
ROC-AUC ≈ 0.85, PR-AUC ≈ 0.74; при 90% precision recall растёт с ~1.5% (логрег) до ~15%.
Сверху `SimpleImputer(median)` – чтобы пропуск в живой форме не ронял инференс.

Итог – бандл `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` в
`backend/models/wine.joblib`; его читает реестр (`app/repositories/model_registry.py`), форма
строится из `features`. Рядом – `wine.model_card.json` с метриками и провенансом.
Этот файл – ЕДИНСТВЕННЫЙ источник истины для описания модели Wine (`ModelInfo`).

Запуск (из каталога backend, в .venv):
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Колонка-таргет и порог «хорошего» вина.
TARGET_COL = "quality"
GOOD_THRESHOLD = 7
# Бинарная целевая -> человекочитаемые метки классов (их увидит UI).
GOOD_LABEL, BAD_LABEL = "Good", "Standard"
POSITIVE_LABEL = GOOD_LABEL

# Метаописание 11 физико-химических признаков (все числовые). Примеры – медианы датасета
# (белое вино), чтобы «Try an example» подставлял осмысленную реальную строку.
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

# Данные лежат в самом репо: training/ -> backend/ -> repo-root(parents[2]) -> data/wine.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "wine" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "wine.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Прочитать датасет, убрать дубли и строки без таргета."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def to_labels(quality: pd.Series) -> pd.Series:
    """Балл качества -> строковый класс Good/Standard (порог >= GOOD_THRESHOLD)."""
    return pd.Series(
        np.where(quality.to_numpy() >= GOOD_THRESHOLD, GOOD_LABEL, BAD_LABEL), index=quality.index
    )


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec для формы: все 11 признаков – числовые (лейбл/единица/пример из FEATURE_META)."""
    return [
        FeatureSpec(
            name=name, type="number", label=meta["label"],
            unit=meta["unit"], example=meta["example"],
        )
        for name, meta in FEATURE_META.items()
    ]


def build_pipeline() -> Pipeline:
    """Импьютер + ExtraTrees (бэггинг деревьев – сильнее линейной модели на этом датасете)."""
    return Pipeline(
        steps=[
            # Импьютер сверху – чтобы пропуск в живой форме не ронял инференс (в данных NaN нет).
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


def compute_metrics(y_bin: np.ndarray, proba_good: np.ndarray) -> dict[str, Any]:
    """Метрики по 5-fold OOF-предсказаниям – всё из одного честного источника, порог 0.5.

    Для несбалансированной бинарной задачи главные метрики – precision/recall/F1 по классу Good и
    ранжирующие ROC-AUC/PR-AUC; accuracy держим в карточке, но в заголовок НЕ выносим (обманчива:
    ~64% даёт даже модель, всегда предсказывающая Standard).
    """
    y_pred = (proba_good >= 0.5).astype(int)  # 1 = Good
    return {
        "precision": float(precision_score(y_bin, y_pred, zero_division=0)),
        "recall": float(recall_score(y_bin, y_pred)),
        "f1": float(f1_score(y_bin, y_pred)),
        "roc_auc": float(roc_auc_score(y_bin, proba_good)),
        "pr_auc": float(average_precision_score(y_bin, proba_good)),
        "accuracy": float(accuracy_score(y_bin, y_pred)),
        # Матрица ошибок в порядке [Standard(0), Good(1)].
        "confusion_matrix": confusion_matrix(y_bin, y_pred, labels=[0, 1]).tolist(),
        "labels": [BAD_LABEL, GOOD_LABEL],
    }


def build_model_info(specs: list[FeatureSpec], metrics: dict[str, Any], n_rows: int) -> ModelInfo:
    """Карточка модели для реестра/формы (is_stub=False – бейдж «demo» исчезнет)."""
    # Заголовок – самые релевантные для задачи метрики класса Good: precision + recall + ROC-AUC
    # (всё по 5-fold OOF). Accuracy НЕ показываем – для несбаланс. бинарной задачи обманчива.
    prec = metrics["precision"]
    rec = metrics["recall"]
    auc = metrics["roc_auc"]
    description = (
        f"Good-wine (quality ≥ 7) classifier – precision {prec:.0%}, recall {rec:.0%}, "
        f"ROC-AUC {auc:.2f} (extra-trees on {n_rows:,} wines)"
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Wine (качество).")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    y = to_labels(df[TARGET_COL])
    y_bin = (y == POSITIVE_LABEL).astype(int).to_numpy()
    good_share = float(y_bin.mean())
    print(f"Обучение: {len(FEATURE_COLUMNS)} фич, ExtraTrees, строк={len(df)}, Good={good_share:.1%}")

    # Оценка – по 5-fold OOF на ВСЕХ данных (устойчивее одиночного holdout из 345 строк, без утечки:
    # каждая строка предсказывается моделью, обученной на других фолдах). proba[:,1] = класс Good.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_good = cross_val_predict(
        build_pipeline(), x, y_bin, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    metrics = compute_metrics(y_bin, oof_good)
    metrics["n_rows"] = len(df)
    metrics["eval"] = "5-fold OOF, threshold 0.5"
    print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
          f"accuracy={metrics['accuracy']:.3f}")

    # Финальная модель – на всех данных (строковые классы Good/Standard для UI).
    final_model = build_pipeline()
    final_model.fit(x, y)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ артефакт: {args.out}  ({size_mb:.2f} МБ)")

    # Model card (провенанс + метрики) рядом с артефактом; реестр читает только *.joblib.
    card = {
        "name": "wine",
        "task": "classification",
        "target": f"quality >= {GOOD_THRESHOLD} ({GOOD_LABEL} vs {BAD_LABEL})",
        "classes": [BAD_LABEL, GOOD_LABEL],
        "estimator": "ExtraTreesClassifier",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Только хвост пути – без абсолютного пути с юзернеймом (не тащим PII в репо).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("wine.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
