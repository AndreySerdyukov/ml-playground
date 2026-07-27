"""Роутер инференса: список моделей и предсказание.

Роутер тонкий: только достаёт сервис из состояния приложения, ловит доменные
исключения и переводит их в HTTP-ответы. Никакой бизнес-логики здесь нет.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.predict import ModelInfo, PredictRequest, PredictResponse
from app.services.inference import (
    InferenceService,
    InvalidFeaturesError,
    ModelNotFoundError,
)

router = APIRouter(prefix="/api", tags=["inference"])


def _service(request: Request) -> InferenceService:
    """Достать singleton-сервис из app.state (положен в main.create_app)."""
    return request.app.state.inference_service


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request) -> list[ModelInfo]:
    """Список доступных моделей с описанием фич (для построения формы)."""
    return _service(request).list_models()


@router.post("/models/{model_name}/predict", response_model=PredictResponse)
def predict(model_name: str, payload: PredictRequest, request: Request) -> PredictResponse:
    """Предсказание выбранной моделью по переданным фичам."""
    try:
        return _service(request).predict(model_name, payload.features)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Модель не найдена: {exc}") from exc
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
