"""Точка входа FastAPI: сборка приложения, CORS, реестр моделей.

Реестр читается один раз на старте и кладётся в состояние приложения вместе с
сервисом инференса — так api-слой получает готовую бизнес-логику без глобалей.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, predict
from app.config import get_settings
from app.repositories.model_registry import ModelRegistry
from app.services.inference import InferenceService


def create_app() -> FastAPI:
    """Фабрика приложения (удобно для тестов и переиспользования)."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    registry = ModelRegistry(settings.models_dir)
    registry.load()
    app.state.inference_service = InferenceService(registry)

    app.include_router(health.router)
    app.include_router(predict.router)
    return app


app = create_app()
