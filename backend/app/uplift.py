"""S-learner (Treatment Dummy) for uplift modeling - serve-side wrapper.

A single trained classification model on features `[profile + treatment_flg]`. Uplift for a customer =
`P(purchase | SMS sent) - P(purchase | SMS not sent)` - two runs of the same classifier with the
treatment flag substituted in. The form asks only for the customer profile; the treatment flag itself is not entered.

The class lives in the `app` package so that joblib unpickles the artifact when the registry loads it (the same reason as
for StubPredictor). The interface is sklearn-compatible: `predict(X)` returns the uplift vector; additionally
`predict_scenarios(X)` returns both probabilities - shown by the "rich" result card.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SoloModelUplift:
    """Wrapper around a trained classifier: uplift as the difference of two treatment scenarios."""

    is_stub = False
    # Task for the registry/UI - a numeric uplift score (regression). The category flags the "rich" card.
    task = "regression"

    def __init__(
        self, estimator: Any, feature_names: list[str], treatment_col: str = "treatment_flg"
    ) -> None:
        # estimator - a trained sklearn Pipeline on columns [feature_names + treatment_col].
        self.estimator = estimator
        # Profile features (in form order), WITHOUT the treatment column.
        self.feature_names = list(feature_names)
        self.treatment_col = treatment_col

    def _proba_at(self, X: pd.DataFrame, flag: int) -> np.ndarray:
        """P(target=1) with the treatment flag substituted in. We take exactly the profile features (dropping extra
        columns) and substitute the flag - the wrapper contract is explicit, and inference does not depend on junk in the input."""
        frame = X[self.feature_names].copy()
        frame[self.treatment_col] = flag
        return np.asarray(self.estimator.predict_proba(frame)[:, 1], dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Uplift score: P(purchase|treatment) - P(purchase|control)."""
        return np.asarray(self._proba_at(X, 1) - self._proba_at(X, 0), dtype=float)

    def predict_scenarios(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """A pair of vectors (P purchase with SMS, P purchase without SMS) - for the result card."""
        return self._proba_at(X, 1), self._proba_at(X, 0)
