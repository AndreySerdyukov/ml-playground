"""Обучение и экспорт РЕАЛЬНОЙ модели «Cars» для ML Playground.

Исходные ноутбуки – в `notebooks/cars/`, данные (train/test) – в `data/cars/` (в самом репо).
Препроцессинг в духе исходного ноутбука:
    - числовые: SimpleImputer(median)
    - категориальные: SimpleImputer(constant "Unknown") + OneHotEncoder(handle_unknown="ignore")
    - таргет priceUSD учится в лог-пространстве (log1p/expm1 через TransformedTargetRegressor):
      это оптимизирует ОТНОСИТЕЛЬНУЮ ошибку (под стать MAPE) и гарантирует положительную цену.
Модель – компактный HistGradientBoostingRegressor: реальные веса, артефакт <1 МБ, обучение секунды.

Домен демо – ОБЫЧНЫЕ б/у авто: строки `for parts`/`with damage` и абсурдные цена/пробег выкинуты
(они дают дикую относительную ошибку и не интересны как «оценка машины»). Поэтому `condition`
из признаков убран (в домене он константа). Метрика в карточке – МЕДИАННАЯ ошибка и доля
попаданий (среднее MAPE раздувает дешёвый хвост; потолок этих фич по MAPE ~14%, см. progress).

Итог – бандл `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` в
`backend/models/cars.joblib`; его читает реестр (`app/repositories/model_registry.py`),
форма строится из `features`. Рядом – `cars.model_card.json` с метриками и провенансом.
Этот файл – ЕДИНСТВЕННЫЙ источник истины для описания модели Cars (`ModelInfo`).

Запуск (из каталога backend, в .venv):
    python training/cars.py                 # prod-набор, чистый домен (по умолчанию)
    python training/cars.py --feature-set A --full   # все 11 фич, без фильтра домена
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

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Колонка-таргет в датасете (в UI отображается как "price").
TARGET_COL = "priceUSD"
# Значение, которым imputer заполняет пропуски категорий (реальная обученная категория).
UNKNOWN_FILL = "Unknown"

# Демо-домен: обычные б/у авто на ходу, без утиля/битых и абсурдных значений.
DOMAIN_CONDITION = "with mileage"
PRICE_MIN, PRICE_MAX = 500, 120_000
MILEAGE_MAX = 700_000

# Метаописание всех потенциальных колонок: тип поля + человекочитаемый лейбл/единица/пример.
# `choices` для категорий НЕ хардкодим – берём из данных (гарантия совпадения с обученным OHE).
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

# Наборы признаков. Порядок = порядок полей формы.
FEATURE_SETS: dict[str, list[str]] = {
    # prod (по умолчанию) – 10 фич без condition (домен отфильтрован на "with mileage").
    "prod": [
        "company", "model", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "color", "vehicle_size_class",
    ],
    # A – полное воспроизведение: все 11 признаков (эксперимент, обычно с --full).
    "A": [
        "company", "model", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "condition", "color", "vehicle_size_class",
    ],
    # B – курированная: без form-killers model/color.
    "B": [
        "company", "year", "mileage(km)", "volume(cm3)", "fuel",
        "transmission", "drive_unit", "vehicle_size_class",
    ],
    # C – как в исходной заглушке: ровно 6 полей.
    "C": ["company", "year", "mileage(km)", "fuel", "volume(cm3)", "transmission"],
}

# Данные лежат в самом репо: training/ -> backend/ -> repo-root(parents[2]) -> data/cars.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "cars" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "cars.joblib"


def load_data(csv_path: Path, clean_domain: bool = True) -> pd.DataFrame:
    """Прочитать датасет, убрать дубли/без цены; опц. сузить до демо-домена обычных авто."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    # Отсекаем нулевые/битые цены: и модель портят, и MAPE делают бесконечной.
    df = df[df[TARGET_COL] > 0]
    if clean_domain:
        df = df[
            (df["condition"] == DOMAIN_CONDITION)
            & df[TARGET_COL].between(PRICE_MIN, PRICE_MAX)
            & df["mileage(km)"].between(1, MILEAGE_MAX)
        ]
    return df.reset_index(drop=True)


def build_feature_specs(df: pd.DataFrame, columns: list[str]) -> list[FeatureSpec]:
    """Собрать FeatureSpec для формы; choices категорий берём из уникальных значений данных."""
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
            # Если в колонке были пропуски – imputer обучил категорию UNKNOWN_FILL,
            # даём её отдельной валидной опцией («не знаю»).
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
    """Препроцессинг + регрессор; таргет учится в лог-пространстве (относительная ошибка)."""
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
    else:  # pragma: no cover - защита от опечатки в аргументе
        raise ValueError(f"Неизвестный estimator: {estimator_name}")

    regressor = Pipeline(steps=[("preproc", preproc), ("estimator", estimator)])
    # Лог-таргет: MSE в лог-пространстве ≈ относительная ошибка (под MAPE) + цена всегда > 0.
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )


def evaluate(
    model: TransformedTargetRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Метрики на holdout: среднее MAPE, МЕДИАННАЯ ошибка, доля попаданий, MAE."""
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
    """Карточка модели для реестра/формы (is_stub=False – бейдж «demo» исчезнет)."""
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
        # Типичная относительная ошибка (median APE) – UI рисует диапазон вокруг оценки.
        typical_error_pct=round(median, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Cars.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--estimator", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="prod")
    parser.add_argument(
        "--full", action="store_true", help="не сужать домен (весь датасет, включая утиль/битые)"
    )
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    clean_domain = not args.full
    df = load_data(args.train_csv, clean_domain=clean_domain)
    columns = FEATURE_SETS[args.feature_set]
    specs = build_feature_specs(df, columns)
    num_features = [s.name for s in specs if s.type == "number"]
    cat_features = [s.name for s in specs if s.type == "category"]

    x = df[columns]
    y = df[TARGET_COL]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    domain = "обычные авто" if clean_domain else "весь датасет"
    print(
        f"Обучение: feature-set={args.feature_set} ({len(columns)} фич), estimator={args.estimator}, "
        f"домен={domain}, строк={len(df)} (train={len(x_train)}, test={len(x_test)})"
    )

    # Оценка на holdout.
    eval_model = build_pipeline(num_features, cat_features, args.estimator)
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(
        f"  MAPE(mean)={metrics['mape']:.3f}  median APE={metrics['median_ape']:.3f}  "
        f"within10%={metrics['within_10pct']:.0%}  MAE={metrics['mae']:.0f} USD"
    )

    # Финальная модель – на всех данных.
    final_model = build_pipeline(num_features, cat_features, args.estimator)
    final_model.fit(x, y)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ артефакт: {args.out}  ({size_mb:.2f} МБ)")

    # Model card (провенанс + метрики) рядом с артефактом; реестр читает только *.joblib.
    card = {
        "name": "cars",
        "task": "regression",
        "target": TARGET_COL,
        "estimator": args.estimator,
        "feature_set": args.feature_set,
        "features": columns,
        "clean_domain": clean_domain,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Только хвост пути – без абсолютного пути с юзернеймом (не тащим PII в репо).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("cars.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
