"""Конфигурация приложения через окружение (pydantic-settings).

Секреты и параметры среды берём только из окружения/`.env`, никогда не хардкодим.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса. Значения переопределяются переменными окружения."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # Человекочитаемое имя приложения (в /health и заголовках).
    app_name: str = "ml-serving-app"
    # Каталог с обученными артефактами моделей (*.joblib).
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    # Разрешённые CORS-источники (фронтенд). Список через запятую в APP_CORS_ORIGINS.
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Синглтон настроек (кэшируется на процесс)."""
    return Settings()
