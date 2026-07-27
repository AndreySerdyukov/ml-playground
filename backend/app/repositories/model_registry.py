"""Реестр моделей: грузит артефакты `*.joblib` из каталога `models/`.

Формат артефакта – dict:
    {"model": <fitted sklearn estimator/pipeline>, "meta": {...ModelInfo...}}

Это data-слой приложения: единственное место, которое знает, откуда берутся модели.
Бизнес-логика инференса (`services/`) реестр только использует, но не знает про файлы.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.schemas.predict import ModelInfo


@dataclass(frozen=True)
class LoadedModel:
    """Загруженная модель: сам estimator + её метаописание."""

    estimator: Any
    info: ModelInfo


class ModelRegistry:
    """Хранит модели в памяти, лениво читая их с диска при инициализации."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models: dict[str, LoadedModel] = {}

    def load(self) -> None:
        """Прочитать все `*.joblib` из каталога моделей в память."""
        self._models.clear()
        if not self._models_dir.exists():
            return
        for path in sorted(self._models_dir.glob("*.joblib")):
            bundle = joblib.load(path)
            info = ModelInfo.model_validate(bundle["meta"])
            self._models[info.name] = LoadedModel(estimator=bundle["model"], info=info)

    def list_infos(self) -> list[ModelInfo]:
        """Список описаний всех доступных моделей."""
        return [m.info for m in self._models.values()]

    def get(self, name: str) -> LoadedModel | None:
        """Модель по имени либо None, если не найдена."""
        return self._models.get(name)
