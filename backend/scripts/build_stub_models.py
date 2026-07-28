"""Стаб-фабрика табличных моделей для веб-оболочки ML Playground.

Все 6 моделей (cars / loans / bayesian / wine / diamonds / uplift) теперь на РЕАЛЬНЫХ весах и
собираются каждая в своём `training/<name>.py`. Стаб-заглушек не осталось → `MODELS` пуст.

Файл сохранён как точка входа для будущих демо-заглушек: добавить кортеж
`(ModelInfo(..., is_stub=True), StubPredictor(...))` в `MODELS`, и артефакт
`backend/models/<name>.joblib` соберётся тем же способом (позже заменяется обученным estimator'ом
с тем же интерфейсом — реестр/фронт не трогаем).

Запуск (из каталога backend):  python scripts/build_stub_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

# Делаем пакет `app` импортируемым при запуске файла напрямую.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import ModelInfo
from app.stub_predictor import StubPredictor

# Описание стаб-моделей: (ModelInfo с is_stub=True) + параметры StubPredictor.
# Сейчас пуст — все модели реальны (см. training/<name>.py). Пример заполнения — в истории git.
MODELS: list[tuple[ModelInfo, StubPredictor]] = []


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for info, predictor in MODELS:
        path = out_dir / f"{info.name}.joblib"
        joblib.dump({"model": predictor, "meta": info.model_dump()}, path)
        print(f"  ✓ {info.name:10s} → {path.name}")
    print(f"Собрано заглушек: {len(MODELS)} (все модели на реальных весах)")


if __name__ == "__main__":
    main()
