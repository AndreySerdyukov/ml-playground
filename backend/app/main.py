"""FastAPI entry point: application assembly, CORS, model registry.

The registry is read once at startup and stored in the application state together with
the inference service - this way the api layer gets ready-made business logic without globals.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, predict
from app.config import get_settings
from app.repositories.model_registry import ModelRegistry
from app.services.inference import InferenceService


def create_app() -> FastAPI:
    """Application factory (convenient for tests and reuse)."""
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
