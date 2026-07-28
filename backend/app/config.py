"""Application configuration via the environment (pydantic-settings).

Secrets and environment parameters are taken only from the environment/`.env`, never hardcoded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings. Values are overridden by environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # Human-readable application name (in /health and headers).
    app_name: str = "ml-serving-app"
    # Directory with trained model artifacts (*.joblib).
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    # Allowed CORS origins (frontend). Override via APP_CORS_ORIGINS as a JSON array,
    # e.g. APP_CORS_ORIGINS='["https://example.com"]' (pydantic-settings parses list fields as JSON).
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Settings singleton (cached per process)."""
    return Settings()
