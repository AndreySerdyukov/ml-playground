"""Собирает стаб-артефакты 6 табличных моделей для веб-оболочки ML Playground.

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

from app.schemas.predict import FeatureSpec, ModelInfo  # noqa: E402
from app.stub_predictor import StubPredictor  # noqa: E402


def num(name: str, label: str, example: float, unit: str | None = None) -> FeatureSpec:
    """Числовое поле формы."""
    return FeatureSpec(name=name, type="number", label=label, example=example, unit=unit)


def cat(name: str, label: str, choices: list[str], example: str) -> FeatureSpec:
    """Категориальное поле формы (выпадающий список)."""
    return FeatureSpec(name=name, type="category", label=label, choices=choices, example=example)


# Описание 6 моделей: (ModelInfo без is_stub) + параметры StubPredictor.
MODELS: list[tuple[ModelInfo, StubPredictor]] = [
    (
        ModelInfo(
            name="wine",
            task="classification",
            target="quality",
            category="Classification",
            emoji="",
            is_stub=True,
            description="Predict wine quality (score 3–8) from physicochemical properties.",
            features=[
                num("fixed acidity", "Fixed acidity", 7.4, "g/L"),
                num("volatile acidity", "Volatile acidity", 0.70, "g/L"),
                num("citric acid", "Citric acid", 0.0, "g/L"),
                num("residual sugar", "Residual sugar", 1.9, "g/L"),
                num("chlorides", "Chlorides", 0.076, "g/L"),
                num("free sulfur dioxide", "Free SO₂", 11, "mg/L"),
                num("total sulfur dioxide", "Total SO₂", 34, "mg/L"),
                num("density", "Density", 0.9978, "g/cm³"),
                num("pH", "pH", 3.51),
                num("sulphates", "Sulphates", 0.56, "g/L"),
                num("alcohol", "Alcohol", 9.4, "% vol"),
            ],
        ),
        StubPredictor(task="classification", classes=["3", "4", "5", "6", "7", "8"]),
    ),
    (
        ModelInfo(
            name="diamonds",
            task="regression",
            target="price",
            target_unit="USD",
            category="Regression",
            emoji="",
            is_stub=True,
            description="Estimate a diamond's sale price from its cut, colour and measurements.",
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
    (
        ModelInfo(
            name="cars",
            task="regression",
            target="price",
            target_unit="USD",
            category="Regression",
            emoji="",
            is_stub=True,
            description="Estimate a used car's price from make, year, mileage and engine.",
            features=[
                cat("company", "Make", ["Toyota", "BMW", "Mercedes-Benz", "Audi", "Volkswagen", "Ford", "Renault", "Kia"], "Toyota"),
                num("year", "Year", 2015),
                num("mileage(km)", "Mileage", 120000, "km"),
                cat("fuel", "Fuel", ["petrol", "diesel", "electro"], "petrol"),
                num("volume(cm3)", "Engine volume", 1600, "cm³"),
                cat("transmission", "Transmission", ["mechanics", "auto"], "auto"),
            ],
        ),
        StubPredictor(task="regression", base=1500.0, scale=0.05),
    ),
    (
        ModelInfo(
            name="bayesian",
            task="regression",
            target="wage",
            target_unit="log $/hr",
            category="Regression",
            emoji="",
            is_stub=True,
            description="Bayesian wage model – predict hourly wage from experience and schooling.",
            features=[
                num("Years", "Labour-market experience", 3, "yrs"),
                num("School", "Years of schooling", 12, "yrs"),
                cat("Union", "Union member", ["No", "Yes"], "No"),
                cat("Married", "Married", ["No", "Yes"], "Yes"),
            ],
        ),
        StubPredictor(task="regression", base=1.2, scale=0.05),
    ),
    (
        ModelInfo(
            name="loans",
            task="classification",
            target="default risk",
            category="Classification",
            emoji="",
            is_stub=True,
            description="Credit-default classifier (illustrative fields – source dataset is anonymised).",
            features=[
                num("annual_income", "Annual income", 45000, "USD"),
                num("loan_amount", "Loan amount", 12000, "USD"),
                cat("term_months", "Term", ["12", "24", "36", "60"], "36"),
                num("employment_years", "Employment length", 4, "yrs"),
                num("existing_debts", "Existing debts", 5000, "USD"),
                cat("purpose", "Purpose", ["car", "education", "home", "business", "other"], "car"),
            ],
        ),
        StubPredictor(task="classification", classes=["Repaid", "Default"]),
    ),
    (
        ModelInfo(
            name="uplift",
            task="regression",
            target="uplift score",
            category="Uplift",
            emoji="",
            is_stub=True,
            description="Uplift model – predicted incremental effect of a marketing treatment.",
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
