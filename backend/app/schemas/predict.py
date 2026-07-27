"""Схемы вход/выход для инференса и описания моделей."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Тип задачи модели (uplift моделируем как регрессию — числовой score).
TaskType = Literal["regression", "classification"]


class FeatureSpec(BaseModel):
    """Описание одной входной фичи (для автогенерации формы на фронте)."""

    name: str
    # Тип поля для UI: число или категория (выпадающий список).
    type: Literal["number", "category"]
    # Допустимые значения для категориальной фичи.
    choices: list[str] | None = None
    # Человекочитаемая подпись поля.
    label: str | None = None
    # Единица измерения (для подписи), напр. "USD", "km".
    unit: str | None = None
    # Значение по умолчанию для кнопки «Try an example».
    example: float | str | None = None


class ModelInfo(BaseModel):
    """Публичное описание модели в реестре (карточка каталога + форма)."""

    name: str
    task: TaskType
    target: str
    description: str = ""
    features: list[FeatureSpec] = Field(default_factory=list)
    # Эмодзи-иконка для карточки каталога.
    emoji: str = ""
    # Отображаемая категория: "Regression" | "Classification" | "Uplift".
    category: str = ""
    # Единица измерения таргета (для карточки результата).
    target_unit: str | None = None
    # Заглушка ли это (веса ещё не подставлены) — UI покажет бейдж «demo».
    is_stub: bool = False


class PredictRequest(BaseModel):
    """Запрос на предсказание: имя модели + значения фич."""

    features: dict[str, Any]


class PredictResponse(BaseModel):
    """Ответ инференса."""

    model_name: str
    task: TaskType
    # Предсказание: число (регрессия) или метка класса (классификация).
    prediction: float | str | int
    # Для классификации — вероятности по классам (если модель их отдаёт).
    probabilities: dict[str, float] | None = None
