"""Бизнес-логика инференса. Чистый слой: без импортов FastAPI.

Отвечает за: валидацию набора фич под конкретную модель, сборку строки-признаков
в правильном порядке колонок и вызов estimator'а. Ошибки – доменные исключения,
которые api-слой транслирует в HTTP-коды.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.repositories.model_registry import ModelRegistry
from app.schemas.predict import ModelInfo, PredictResponse


class ModelNotFoundError(Exception):
    """Запрошена модель, которой нет в реестре."""


class InvalidFeaturesError(Exception):
    """В запросе не хватает обязательных фич модели."""


class InferenceService:
    """Оркестрирует предсказание поверх реестра моделей."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def list_models(self) -> list[ModelInfo]:
        """Описания доступных моделей (для UI и API)."""
        return self._registry.list_infos()

    def predict(self, model_name: str, features: dict[str, object]) -> PredictResponse:
        """Сделать предсказание моделью `model_name` по словарю фич.

        Порядок колонок берём из метаописания модели – pipeline ожидает именно его.
        """
        loaded = self._registry.get(model_name)
        if loaded is None:
            raise ModelNotFoundError(model_name)

        expected = [f.name for f in loaded.info.features]
        missing = [name for name in expected if name not in features]
        if missing:
            raise InvalidFeaturesError(f"Не хватает фич: {', '.join(missing)}")

        row = pd.DataFrame([{name: features[name] for name in expected}])
        estimator = loaded.estimator

        if loaded.info.task == "classification":
            label = estimator.predict(row)[0]
            probabilities = self._extract_proba(estimator, row)
            return PredictResponse(
                model_name=model_name,
                task="classification",
                prediction=self._to_python(label),
                probabilities=probabilities,
            )

        value = float(estimator.predict(row)[0])
        return PredictResponse(model_name=model_name, task="regression", prediction=value)

    @staticmethod
    def _extract_proba(estimator: object, row: pd.DataFrame) -> dict[str, float] | None:
        """Вернуть вероятности по классам, если estimator их поддерживает."""
        if not hasattr(estimator, "predict_proba"):
            return None
        proba = estimator.predict_proba(row)[0]
        classes = [str(c) for c in getattr(estimator, "classes_", range(len(proba)))]
        return {cls: float(p) for cls, p in zip(classes, proba)}

    @staticmethod
    def _to_python(value: object) -> float | int | str:
        """Привести numpy-скаляр к нативному типу Python для сериализации."""
        if isinstance(value, np.generic):
            return value.item()
        return value  # type: ignore[return-value]
