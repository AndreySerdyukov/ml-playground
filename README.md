# ML Playground

**Run trained machine-learning models in the browser.** Pick a model, fill in the inputs, get a
prediction. One super-app hosting six tabular models behind a clean, Apple-inspired UI.

`FastAPI` · `React + TypeScript` · `scikit-learn` · model-agnostic forms · light/dark · single & batch

---

## What it is

Each model ships as a self-contained artifact (`{"model": estimator, "meta": ModelInfo}`). The backend
serves a catalog; the frontend **generates the input form automatically** from each model's feature spec
and renders a result card tailored to the task (a value with a typical-error band for regression, class
probabilities with an interactive decision threshold for classification, a two-scenario lift card for
uplift). Adding or swapping a model needs **no UI code** – the metadata drives everything.

Every model is trained by a standalone script under `backend/training/`, reads its data from `data/`, and
writes the artifact plus a `*.model_card.json` (metrics + provenance). The exploratory notebooks that each
script distils live in `notebooks/`.

## Highlights

- **Six real trained models** (no stubs) spanning regression, classification and uplift.
- **Model-agnostic UI** – forms, dropdowns, examples and result cards are all built from backend metadata.
- **Single and batch prediction** – one form, or upload a CSV/Excel and score every row.
- **Interactive decision threshold** for binary classifiers – the label and precision/recall update live
  as you drag the slider (computed client-side from the returned probabilities).
- **Uplift decision card** – shows the incremental lift in percentage points, the purchase probability
  with and without treatment, and a target / don't-target verdict.
- **Light and dark theme** via CSS variables; strict TypeScript; layered, framework-free business logic.
- **Reproducible** – data and notebooks are in the repo, so any model rebuilds from a clean clone.

## Models

| Model | Predicts | Type | Approach | Held-out quality |
|---|---|---|---|---|
| **Cars** | Used-car price (USD) | Regression | HistGradientBoosting on a log target | median error ≈ 12%, 44% within 10% |
| **Wine** | Wine quality is "good" (≥ 7) | Classification | Extra-Trees + tuned decision threshold | ROC-AUC 0.85, F1 0.70 |
| **Diamonds** | Diamond price (USD) | Regression | HistGradientBoosting (gamma loss) | **MAPE 6.9%** on a held-out test set |
| **Loans** | Credit application approved | Classification | Stacking ensemble (GB + AdaBoost + RF → LogReg) | accuracy 86%, ROC-AUC 0.93 |
| **Uplift** | Incremental effect of a promo SMS | Uplift | S-learner (Treatment Dummy, HistGradientBoosting) | Qini 0.08, uplift@30% 5.7pp vs 3.3pp average |
| **Bayesian** | Hourly wage ($/hr) | Regression | Bayesian ridge on a log target | median error ≈ 26%, MAE $1.98/hr |

Full metrics and provenance for each model are in `backend/models/<name>.model_card.json`; the modeling
write-ups are in `notebooks/<name>/`.

## Architecture

```
frontend (React)  →  api/ (routers)  →  services/ (inference + file ingest)  →  repositories/ (model registry)
```

- **Layered and framework-free.** Business logic in `services/` never imports FastAPI, so it is unit-testable
  in isolation; `repositories/` is the only place that knows artifacts come from `*.joblib` files.
- **Stateless inference** – no database or cache by design.
- **Metadata-driven.** A model's `ModelInfo` (name, task, feature specs, category, error band, threshold
  curve, …) is serialized inside the artifact and is the single source of truth for both serving and the UI.

## Project layout

```
backend/
  app/            FastAPI app: api / services / repositories / schemas, plus serving wrappers
  training/       one script per model – reads data/<name>/, writes models/<name>.joblib + model_card
  models/         built artifacts (*.joblib, gitignored) + *.model_card.json
  scripts/        build_stub_models.py (empty now – all models are real)
data/<name>/      training data + a short README per model
notebooks/<name>/ the research notebooks each training script distils
frontend/         React + TypeScript (Vite, Tailwind, react-router)
```

## Run locally

```bash
# Backend
cd backend
uv venv --python 3.12
uv pip install fastapi "uvicorn[standard]" pydantic pydantic-settings joblib "scikit-learn~=1.9.0" \
  pandas numpy python-multipart openpyxl pytest
# Train the artifacts (they are gitignored; rebuild from the in-repo data):
python training/cars.py && python training/wine.py && python training/diamonds.py \
  && python training/loans.py && python training/uplift.py && python training/bayesian.py
uv run uvicorn app.main:app --reload       # http://127.0.0.1:8000

# Frontend (another terminal)
cd frontend
npm install && npm run dev                 # http://127.0.0.1:5173, /api proxied to :8000
```

## Run in Docker

```bash
docker compose up --build                  # frontend :3000, backend :8000
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/models` | Catalog with per-model feature specs |
| `POST` | `/api/models/{name}/predict` | `{"features": {...}}` → a single prediction |
| `POST` | `/api/models/{name}/predict-batch` | Upload a CSV/Excel, get a prediction per row |

## Development

```bash
cd backend && ruff check . && mypy app && pytest      # lint, type-check, tests
cd frontend && npm run build                          # type-check + production build
```

The scikit-learn version is pinned (`~=1.9`) because a joblib artifact must be unpickled by the same minor
version it was trained with.

## Roadmap

- Feature-importance (SHAP) in the result card.
- Host a live demo (FastAPI on a small VPS/PaaS, frontend as static assets).
- Model artifact delivery: rebuild on deploy from in-repo data (current), commit artifacts, or Git LFS.
