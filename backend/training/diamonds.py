"""Обучение и экспорт РЕАЛЬНОЙ модели «Diamonds» для ML Playground.

Данные (train/test/test_Y_true) – в `data/diamonds/` (в самом репо), метрика задачи – MAPE.

ГЛАВНЫЙ УРОК этого скрипта – корректная работа с выбросами (в исходном ноутбуке она сломана
и раздувала MAPE до 18.68 при групповом лучшем ~7.31):
    - в ноутбуке цикл 1.5*IQR шёл по ВСЕМ числовым колонкам, ВКЛЮЧАЯ таргет
      `total_sales_price`, и вырезал из обучения ~11% строк – весь дорогой хвост (цена > ~6700).
      Настоящий `test.csv` хвост сохраняет → модель не дотягивается до дорогих камней → взрыв MAPE.
    - иногда та же отсечка делалась ДО train_test_split → «чистился» и тестовый кусок (лик,
      оптимистичные 11.70, которые на реальном тесте превращаются в 18.68).
Здесь – как надо: СНАЧАЛА сплит, чистка/импутация живут ВНУТРИ sklearn-pipeline (fit только на
train, transform на остальном → утечки нет по построению), по ТАРГЕТУ строки НЕ фильтруем, а
внешний `test.csv`/`test_Y_true.csv` руками не трогаем до финального скоринга.

Препроцессинг:
    - числовые (size, depth/table %, meas_*): невозможные нули → NaN, затем SimpleImputer(median)
      (аналог IterativeImputer(missing_values=0) из ноутбука, но честно на train).
    - категориальные (color, clarity, cut, symmetry, polish) – упорядоченные грейды →
      OrdinalEncoder с явным порядком (деревьям монотонный код грейда на пользу).
    - таргет учится в лог-пространстве (log1p/expm1) ЛИБО через нативный loss="gamma" – берём
      лучший по внутренней валидации (оба варианта оптимизируют ОТНОСИТЕЛЬНУЮ ошибку под MAPE).
Модель – HistGradientBoostingRegressor (реальные веса, обучение секунды).

Итог – бандл `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` в
`backend/models/diamonds.joblib`; его читает реестр (`app/repositories/model_registry.py`),
форма строится из `features`. Рядом – `diamonds.model_card.json` с метриками и провенансом.
Этот файл – ЕДИНСТВЕННЫЙ источник истины для описания модели Diamonds (`ModelInfo`).

Запуск (из каталога backend, в .venv):
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

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Колонка-таргет в датасете (в UI отображается как "price").
TARGET_COL = "total_sales_price"

# Упорядоченные грейды (best → worst). Один список на признак:
#   - идёт в OrdinalEncoder как явный порядок категорий (монотонный код);
#   - и в FeatureSpec.choices формы (choices == обученные категории, unknown исключён).
GRADE_ORDER: dict[str, list[str]] = {
    "color": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
    "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"],
    "cut": ["Excellent", "Very Good"],
    "symmetry": ["Excellent", "Very Good"],
    "polish": ["Excellent", "Very Good"],
}

# Метаописание колонок: тип поля + лейбл/единица/пример. Порядок = порядок полей формы.
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

# Данные лежат в самом репо: training/ -> backend/ -> repo-root(parents[2]) -> data/diamonds.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "diamonds"
DEFAULT_TRAIN_CSV = DATA_DIR / "train.csv"
DEFAULT_TEST_CSV = DATA_DIR / "test.csv"
DEFAULT_TEST_Y_CSV = DATA_DIR / "test_Y_true.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "diamonds.joblib"


def load_train(csv_path: Path) -> pd.DataFrame:
    """Прочитать train, убрать дубли и строки без цены. По ТАРГЕТУ выбросы НЕ режем."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec для формы; choices категорий = обученные грейды из GRADE_ORDER (в порядке качества)."""
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
    """Препроцессинг + HGB. kind: 'log' (лог-таргет + squared_error) | 'gamma' (нативный loss)."""
    # Невозможные нули в числовых фичах (замеры/проценты == 0) чиним медианой train, затем
    # второй imputer страхует от пустого поля в живой форме (NaN). Оба – встроенные (пиклятся).
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
        # Лог-таргет: MSE в лог-пространстве ≈ относительная ошибка (под MAPE) + цена всегда > 0.
        return TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1
        )
    if kind == "gamma":
        # Gamma-loss моделирует положительный скошенный таргет с ~постоянным CV → тоже под MAPE.
        estimator = HistGradientBoostingRegressor(loss="gamma", **common)
        return Pipeline(steps=[("preproc", preproc), ("estimator", estimator)])
    raise ValueError(f"Неизвестный kind: {kind}")  # pragma: no cover


def evaluate(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Метрики: MAPE (среднее), медианная ошибка, доли попаданий, MAE, max error."""
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
    """Карточка модели для реестра/формы (is_stub=False – бейдж «demo» исчезнет)."""
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
        # Типичная относит. ошибка (median APE на реальном тесте) – UI рисует диапазон вокруг оценки.
        typical_error_pct=round(median, 3),
    )


def _fmt(m: dict[str, float]) -> str:
    return (
        f"MAPE={m['mape']:.4f}  medianAPE={m['median_ape']:.4f}  "
        f"within10%={m['within_10pct']:.0%}  MAE={m['mae']:.0f} USD"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Diamonds.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--test-y-csv", type=Path, default=DEFAULT_TEST_Y_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--kind", choices=["auto", "log", "gamma"], default="auto",
        help="вариант модели: auto = выбрать лучший по внутренней валидации",
    )
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    df = load_train(args.train_csv)
    specs = build_feature_specs()
    x = df[FEATURES]
    y = df[TARGET_COL]

    # 1) СНАЧАЛА сплит. Вся чистка/импутация – внутри pipeline (fit только на train). Утечки нет.
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"train.csv: строк={len(df)} (train={len(x_train)}, val={len(x_val)}); фич={len(FEATURES)}")

    # 2) Выбор варианта модели по внутренней валидации (или заданный явно).
    kinds = ["log", "gamma"] if args.kind == "auto" else [args.kind]
    val_scores: dict[str, dict[str, float]] = {}
    for kind in kinds:
        m = build_pipeline(kind)
        m.fit(x_train, y_train)
        val_scores[kind] = evaluate(m, x_val, y_val)
        print(f"  [{kind:5s}] holdout: {_fmt(val_scores[kind])}")
    best_kind = min(val_scores, key=lambda k: val_scores[k]["mape"])
    print(f"  → выбран вариант: {best_kind}")

    # 3) Честный внешний скоринг (сопоставим с 18.68 / групповым 7.31): обучаем на ПОЛНОМ train.csv,
    #    предсказываем внешний test.csv, сверяем с test_Y_true.csv. Тест руками не трогали.
    final_model = build_pipeline(best_kind)
    final_model.fit(x, y)
    test_metrics: dict[str, float] = {}
    if args.test_csv.exists() and args.test_y_csv.exists():
        x_test = pd.read_csv(args.test_csv)[FEATURES]
        y_test = pd.read_csv(args.test_y_csv)[TARGET_COL]
        test_metrics = evaluate(final_model, x_test, y_test)
        print(f"  ВНЕШНИЙ ТЕСТ (n={len(x_test)}): {_fmt(test_metrics)}  maxErr={test_metrics['max_error']:.0f}")
    else:
        print("  ⚠ test.csv/test_Y_true.csv не найдены – внешний скоринг пропущен, беру holdout-метрики")
        test_metrics = val_scores[best_kind]

    # 4) Экспорт. Отгружаем модель, обученную на train.csv (на метках теста не учимся).
    info = build_model_info(specs, test_metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ артефакт: {args.out}  ({size_mb:.2f} МБ)")

    # Model card (провенанс + метрики) рядом с артефактом; реестр читает только *.joblib.
    card = {
        "name": "diamonds",
        "task": "regression",
        "target": TARGET_COL,
        "estimator": f"HistGradientBoostingRegressor ({best_kind})",
        "features": FEATURES,
        "metrics_holdout": val_scores.get(best_kind, {}),
        "metrics_external_test": test_metrics,
        "sklearn_version": sklearn.__version__,
        # Только хвост пути – без абсолютного пути с юзернеймом (не тащим PII в репо).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("diamonds.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
