"""S-learner (Treatment Dummy) для uplift-моделирования — serve-side обёртка.

Одна обученная модель классификации на признаках `[профиль + treatment_flg]`. Uplift для клиента =
`P(покупка | SMS отправлен) − P(покупка | SMS не отправлен)` — два прогона того же классификатора с
подставленным флагом лечения. Форма спрашивает только профиль клиента; сам флаг лечения не вводится.

Класс лежит в пакете `app`, чтобы joblib распиклил артефакт при загрузке реестром (та же причина, что
и для StubPredictor). Интерфейс sklearn-совместимый: `predict(X)` возвращает вектор uplift; дополнительно
`predict_scenarios(X)` отдаёт обе вероятности — их показывает «богатая» карточка результата.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SoloModelUplift:
    """Обёртка вокруг обученного классификатора: uplift как разность двух сценариев лечения."""

    is_stub = False
    # Задача для реестра/UI — числовой uplift-скор (регрессия). Категория помечает «богатую» карточку.
    task = "regression"

    def __init__(
        self, estimator: Any, feature_names: list[str], treatment_col: str = "treatment_flg"
    ) -> None:
        # estimator — обученный sklearn Pipeline на колонках [feature_names + treatment_col].
        self.estimator = estimator
        # Профильные фичи (в порядке формы), БЕЗ колонки лечения.
        self.feature_names = list(feature_names)
        self.treatment_col = treatment_col

    def _proba_at(self, X: pd.DataFrame, flag: int) -> np.ndarray:
        """P(target=1) при подставленном флаге лечения. Берём ровно профильные фичи (отсекаем лишние
        колонки) и подставляем флаг — контракт обёртки явный, инференс не зависит от мусора во входе."""
        frame = X[self.feature_names].copy()
        frame[self.treatment_col] = flag
        return np.asarray(self.estimator.predict_proba(frame)[:, 1], dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Uplift-скор: P(покупка|лечение) − P(покупка|контроль)."""
        return np.asarray(self._proba_at(X, 1) - self._proba_at(X, 0), dtype=float)

    def predict_scenarios(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Пара векторов (P покупки при SMS, P покупки без SMS) — для карточки результата."""
        return self._proba_at(X, 1), self._proba_at(X, 0)
