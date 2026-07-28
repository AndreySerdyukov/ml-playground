"""Stub factory for tabular models in the ML Playground web shell.

All 6 models (cars / loans / bayesian / wine / diamonds / uplift) now run on REAL weights and
each is built in its own `training/<name>.py`. No stubs remain → `MODELS` is empty.

The file is kept as an entry point for future demo stubs: add a tuple
`(ModelInfo(..., is_stub=True), StubPredictor(...))` to `MODELS`, and the artifact
`backend/models/<name>.joblib` will be built the same way (later replaced by a trained estimator
with the same interface - we don't touch the registry/frontend).

Run (from the backend directory):  python scripts/build_stub_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

# Make the `app` package importable when running the file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import ModelInfo
from app.stub_predictor import StubPredictor

# Stub model definitions: (ModelInfo with is_stub=True) + StubPredictor parameters.
# Currently empty - all models are real (see training/<name>.py). An example of how to fill it is in git history.
MODELS: list[tuple[ModelInfo, StubPredictor]] = []


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for info, predictor in MODELS:
        path = out_dir / f"{info.name}.joblib"
        joblib.dump({"model": predictor, "meta": info.model_dump()}, path)
        print(f"  ✓ {info.name:10s} → {path.name}")
    print(f"Stubs built: {len(MODELS)} (all models on real weights)")


if __name__ == "__main__":
    main()
