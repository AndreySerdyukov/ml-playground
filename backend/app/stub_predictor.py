"""Stub predictor for the web shell (model weights are not plugged in yet).

Produces a plausible, **input-deterministic** response so that the whole
UI -> API -> prediction flow works end-to-end BEFORE the real weights exist. Later this object
is replaced by a trained sklearn estimator (the same `predict`/`predict_proba`/
`classes_` interface), with no need to change the frontend/backend.

The class lives in the `app` package so that joblib can unpickle the artifact when the registry loads it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StubPredictor:
    """Mini-predictor with a sklearn-like interface. Marked `is_stub = True`."""

    is_stub = True

    def __init__(
        self,
        task: str,
        classes: list[str] | None = None,
        base: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.task = task
        # sklearn-style attribute - the api reads class labels from it.
        self.classes_ = list(classes) if classes else None
        self._base = base
        self._scale = scale

    @staticmethod
    def _score(row: dict[str, object]) -> float:
        """Stable numeric signal derived from a feature row."""
        total = 0.0
        for value in row.values():
            if isinstance(value, bool):
                total += 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                total += float(value)
            else:
                # Categorical - stable hash into a small range.
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
