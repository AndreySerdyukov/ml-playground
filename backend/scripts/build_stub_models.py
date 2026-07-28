"""Собирает стаб-артефакты табличных моделей для веб-оболочки ML Playground.

Каждый артефакт – `{"model": StubPredictor(...), "meta": ModelInfo(...).model_dump()}`
в `backend/models/<name>.joblib`. Признаки курированы (чистые лейблы/единицы/примеры),
чтобы форма выглядела аккуратно. Реальные веса подставим позже, заменив StubPredictor
обученным estimator'ом с тем же интерфейсом.

Запуск (из каталога backend):  python scripts/build_stub_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

# Делаем пакет `app` импортируемым при запуске файла напрямую.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo
from app.stub_predictor import StubPredictor


def num(name: str, label: str, example: float, unit: str | None = None) -> FeatureSpec:
    """Числовое поле формы."""
    return FeatureSpec(name=name, type="number", label=label, example=example, unit=unit)


def cat(name: str, label: str, choices: list[str], example: str) -> FeatureSpec:
    """Категориальное поле формы (выпадающий список)."""
    return FeatureSpec(name=name, type="category", label=label, choices=choices, example=example)


# Описание стаб-моделей: (ModelInfo без is_stub) + параметры StubPredictor.
# Реальные модели (cars/wine/bayesian/loans) здесь НЕ описаны – они собираются
# отдельно в training/<name>.py; ниже остаются только настоящие заглушки.
MODELS: list[tuple[ModelInfo, StubPredictor]] = [
    # wine – РЕАЛЬНАЯ обученная модель, собирается отдельно в training/wine.py
    # (не стаб). Здесь запись убрана намеренно, чтобы стаб-фабрика не перетирала
    # backend/models/wine.joblib при перегенерации остальных заглушек.
    (
        ModelInfo(
            name="diamonds",
            task="regression",
            target="price",
            target_unit="USD",
            category="Regression",
            emoji="",
            is_stub=True,
            description="Estimate a diamond's sale price from its cut, colour and measurements",
            features=[
                num("size", "Carat", 0.7, "ct"),
                cat("cut", "Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], "Ideal"),
                cat("color", "Colour", ["D", "E", "F", "G", "H", "I", "J"], "G"),
                cat("clarity", "Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], "VS2"),
                num("depth_percent", "Depth", 61.5, "%"),
                num("table_percent", "Table", 57.0, "%"),
            ],
        ),
        StubPredictor(task="regression", base=300.0, scale=20.0),
    ),
    # cars – РЕАЛЬНАЯ обученная модель, собирается отдельно в training/cars.py
    # (не стаб). Здесь запись убрана намеренно, чтобы стаб-фабрика не перетирала
    # backend/models/cars.joblib при перегенерации остальных заглушек.
    # bayesian – РЕАЛЬНАЯ обученная модель, собирается отдельно в training/bayesian.py
    # (не стаб). Здесь запись убрана намеренно, чтобы стаб-фабрика не перетирала
    # backend/models/bayesian.joblib при перегенерации остальных заглушек.
    # loans – РЕАЛЬНАЯ обученная модель, собирается отдельно в training/loans.py
    # (не стаб). Здесь запись убрана намеренно, чтобы стаб-фабрика не перетирала
    # backend/models/loans.joblib при перегенерации остальных заглушек.
    (
        ModelInfo(
            name="uplift",
            task="regression",
            target="uplift score",
            category="Uplift",
            emoji="",
            is_stub=True,
            description="Uplift model – predicted incremental effect of a marketing treatment",
            features=[
                num("recency", "Days since last purchase", 30, "days"),
                num("frequency", "Purchases last year", 6),
                num("monetary", "Average spend", 80, "USD"),
                cat("channel", "Channel", ["email", "web", "phone"], "email"),
                cat("used_discount", "Used discount before", ["No", "Yes"], "Yes"),
            ],
        ),
        StubPredictor(task="regression", base=0.02, scale=0.001),
    ),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for info, predictor in MODELS:
        path = out_dir / f"{info.name}.joblib"
        joblib.dump({"model": predictor, "meta": info.model_dump()}, path)
        print(f"  ✓ {info.name:10s} → {path.name}")
    print(f"Собрано моделей: {len(MODELS)} в {out_dir}")


if __name__ == "__main__":
    main()
