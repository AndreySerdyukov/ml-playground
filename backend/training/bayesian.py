"""Обучение и экспорт РЕАЛЬНОЙ модели «Bayesian» (зарплата) для ML Playground.

Исходный ноутбук – в `notebooks/bayesian/`, данные – в `data/bayesian/` (в самом репо).
Датасет NLS: 545 молодых работников за 1980–1987 (≈4360 строк «работник-год»). Цель – оценить
почасовую зарплату по образованию, опыту, профсоюзу, браку и расе. В ноутбуке это байесовская
линейная регрессия (PyMC, StudentT); финальная модель `model_five` использует признаки
`Union, Expert, School, Married, Black`.

В прод берём её лёгкий эквивалент – sklearn `BayesianRidge` (настоящая байесовская линейная
регрессия: гауссовы приоры на веса, оценка через evidence). Артефакт крошечный, PyMC в рантайме
не нужен, а коэффициенты воспроизводят вывод ноутбука (union-премия ≈ +20%, отдача на образование
≈ +10%/год). Колонка `Wage` – ЛОГАРИФМ зарплаты; учим в лог-пространстве и отдаём `exp(Wage)` –
доллары в час (положительные, интуитивные), тот же приём лог-таргета, что и у Cars.

Итог – бандл `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` в
`backend/models/bayesian.joblib`; его читает реестр (`app/repositories/model_registry.py`),
форма строится из `features`. Рядом – `bayesian.model_card.json` с метриками и провенансом.
Этот файл – ЕДИНСТВЕННЫЙ источник истины для описания модели Bayesian (`ModelInfo`).

Запуск (из каталога backend, в .venv):
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

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo

# Колонка-таргет в датасете – ЛОГАРИФМ почасовой зарплаты.
TARGET_COL = "Wage"

# Метаописание признаков: тип поля + человекочитаемый лейбл/единица/пример.
FEATURE_META: dict[str, dict[str, Any]] = {
    "School": {"kind": "num", "label": "Years of schooling", "example": 12},
    "Expert": {"kind": "num", "label": "Work experience", "unit": "yrs", "example": 5},
    "Union": {"kind": "bin", "label": "Union member", "example": "Yes"},
    "Married": {"kind": "bin", "label": "Married", "example": "Yes"},
    "Black": {"kind": "bin", "label": "Black", "example": "No"},
}
# Порядок = порядок полей формы (как в ноутбуке model_five).
FEATURE_COLUMNS: list[str] = ["School", "Expert", "Union", "Married", "Black"]
# Бинарные колонки: в данных 0/1, в форме показываем как No/Yes (дропдаун).
BINARY_COLUMNS: list[str] = [n for n, m in FEATURE_META.items() if m["kind"] == "bin"]
BINARY_MAP: dict[float, str] = {0.0: "No", 1.0: "Yes"}

# Данные лежат в самом репо: training/ -> backend/ -> repo-root(parents[2]) -> data/bayesian.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "bayesian" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "bayesian.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Прочитать датасет, убрать строки без таргета; бинарные 0/1 → No/Yes (как в форме)."""
    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=[TARGET_COL])
    for col in BINARY_COLUMNS:
        df[col] = df[col].astype(float).map(BINARY_MAP)
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec для формы: School/Expert – числа, Union/Married/Black – дропдаун No/Yes."""
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
    """Препроцессинг + BayesianRidge; таргет ($/час) учится в лог-пространстве (= log-зарплата).

    Числовые (School/Expert) пропускаем как есть; бинарные one-hot'им (drop='if_binary' → 1 колонка).
    `TransformedTargetRegressor(log/exp)`: модель обучается на log($/час) = исходный `Wage`, а predict
    возвращает exp(...) = $/час. Это в точности повторяет лог-регрессию ноутбука, но отдаёт доллары.
    """
    num_features = [n for n in FEATURE_COLUMNS if FEATURE_META[n]["kind"] == "num"]
    preproc = ColumnTransformer(
        transformers=[
            # Импьютер сверху – чтобы пропуск в живой форме не ронял инференс (в данных NaN нет).
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
    """Метрики на holdout ($/час): среднее MAPE, МЕДИАННАЯ ошибка, доля попаданий, MAE."""
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
    """Карточка модели для реестра/формы (is_stub=False – бейдж «demo» исчезнет)."""
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
        # Типичная относит. ошибка (median APE) – UI рисует диапазон вокруг оценки.
        typical_error_pct=round(median, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Bayesian (зарплата).")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    # `Wage` – лог зарплаты; целевую для обучения/оценки держим в $/час (exp), лог делает пайплайн.
    y = np.exp(df[TARGET_COL].astype(float))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Обучение: {len(FEATURE_COLUMNS)} фич, BayesianRidge, строк={len(df)} "
          f"(train={len(x_train)}, test={len(x_test)})")

    # Оценка на holdout.
    eval_model = build_pipeline()
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(f"  median APE={metrics['median_ape']:.3f}  MAPE(mean)={metrics['mape']:.3f}  "
          f"within15%={metrics['within_15pct']:.0%}  MAE=${metrics['mae']:.2f}/hr")

    # Финальная модель – на всех данных.
    final_model = build_pipeline()
    final_model.fit(x, y)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_kb = args.out.stat().st_size / 1024
    print(f"  ✓ артефакт: {args.out}  ({size_kb:.0f} КБ)")

    # Model card (провенанс + метрики) рядом с артефактом; реестр читает только *.joblib.
    card = {
        "name": "bayesian",
        "task": "regression",
        "target": "exp(Wage) = $/hr",
        "estimator": "BayesianRidge + log-target",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Только хвост пути – без абсолютного пути с юзернеймом (не тащим PII в репо).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("bayesian.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
