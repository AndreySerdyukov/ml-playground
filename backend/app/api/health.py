"""Health-check роутер."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Простой liveness-пробник для healthcheck при деплое."""
    return {"status": "ok", "app": get_settings().app_name}
