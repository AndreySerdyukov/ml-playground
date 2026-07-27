"""Заглушка-предиктор для веб-оболочки (веса моделей ещё не подставлены).

Даёт правдоподобный, **детерминированный от входа** ответ, чтобы весь поток
UI → API → предсказание работал end-to-end ДО реальных весов. Позже этот объект
заменяется обученным sklearn-estimator'ом (тот же интерфейс `predict`/`predict_proba`/
`classes_`), и менять фронт/бэк не нужно.

Класс лежит в пакете `app`, чтобы joblib мог распиклить артефакт при загрузке реестром.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StubPredictor:
    """Мини-предиктор с sklearn-подобным интерфейсом. Помечен `is_stub = True`."""

    is_stub = True

    def __init__(
        self,
        task: str,
        classes: list[str] | None = None,
        base: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.task = task
        # Атрибут в стиле sklearn – из него api берёт метки классов.
        self.classes_ = list(classes) if classes else None
        self._base = base
        self._scale = scale

    @staticmethod
    def _score(row: dict[str, object]) -> float:
        """Стабильный числовой сигнал от строки признаков."""
        total = 0.0
        for value in row.values():
            if isinstance(value, bool):
                total += 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                total += float(value)
            else:
                # Категориальное – стабильный хэш в небольшой диапазон.
                total += sum(ord(ch) for ch in str(value)) % 10
        return total

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rows = X.to_dict("records")
        if self.task == "classification" and self.classes_:
            idx = [int(abs(self._score(r))) % len(self.classes_) for r in rows]
            return np.array([self.classes_[i] for i in idx])
        return np.array([self._base + self._scale * self._score(r) for r in rows])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rows = X.to_dict("records")
        n = len(self.classes_ or [])
        out = []
        for r in rows:
            raw = np.array([((self._score(r) + i * 7.0) % 100) + 1.0 for i in range(n)])
            out.append(raw / raw.sum())
        return np.array(out)
