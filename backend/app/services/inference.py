"""Бизнес-логика инференса. Чистый слой: без импортов FastAPI.

Отвечает за: валидацию набора фич под конкретную модель, сборку строки-признаков
в правильном порядке колонок и вызов estimator'а. Ошибки – доменные исключения,
которые api-слой транслирует в HTTP-коды.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.repositories.model_registry import ModelRegistry
from app.schemas.predict import BatchPredictResponse, ModelInfo, PredictResponse


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
        # Uplift-модели (S-learner) дополнительно отдают вероятности исхода в двух сценариях лечения —
        # их показывает «богатая» карточка. Прочие регрессоры этого метода не имеют → scenarios=None.
        scenarios = None
        if hasattr(estimator, "predict_scenarios"):
            p_treat, p_control = estimator.predict_scenarios(row)
            scenarios = {
                "with_treatment": float(p_treat[0]),
                "without_treatment": float(p_control[0]),
            }
        return PredictResponse(
            model_name=model_name, task="regression", prediction=value, scenarios=scenarios
        )

    def predict_batch(
        self, model_name: str, records: list[dict[str, object]]
    ) -> BatchPredictResponse:
        """Предсказание по многим строкам сразу (из загруженного CSV/Excel).

        Возвращает значения фич по строкам + колонку `prediction` (порядок строк сохраняется).
        Числовые фичи приводятся к числам; пропуски и незнакомые категории покрывает пайплайн.
        """
        loaded = self._registry.get(model_name)
        if loaded is None:
            raise ModelNotFoundError(model_name)
        if not records:
            raise InvalidFeaturesError("Нет строк для предсказания")

        expected = [f.name for f in loaded.info.features]
        frame = pd.DataFrame(records)
        missing = [name for name in expected if name not in frame.columns]
        if missing:
            raise InvalidFeaturesError(f"В файле не хватает колонок: {', '.join(missing)}")

        rows_in = frame[expected].copy()
        for feature in loaded.info.features:
            if feature.type == "number":
                rows_in[feature.name] = pd.to_numeric(rows_in[feature.name], errors="coerce")

        preds = loaded.estimator.predict(rows_in)
        is_cls = loaded.info.task == "classification"
        out_rows: list[dict[str, Any]] = []
        for i, record in enumerate(rows_in.to_dict("records")):
            row: dict[str, Any] = {name: self._cell(record[name]) for name in expected}
            # 4 знака: у денежных таргетов хватает, а мелкие uplift-скоры (~0.01–0.05) не схлопываются.
            row["prediction"] = self._to_python(preds[i]) if is_cls else round(float(preds[i]), 4)
            out_rows.append(row)

        return BatchPredictResponse(
            model_name=model_name,
            task=loaded.info.task,
            target=loaded.info.target,
            target_unit=loaded.info.target_unit,
            columns=expected,
            rows=out_rows,
            count=len(out_rows),
        )

    @staticmethod
    def _cell(value: object) -> object:
        """Скаляр ячейки → JSON-совместимый тип (NaN/пропуск → None)."""
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

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
