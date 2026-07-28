# ML Playground

Interactive web app to **run trained ML models in the browser** — pick a model, enter the inputs,
get a prediction. A single super-app hosting several tabular models (wine quality, diamond & car
price, wage, credit risk, uplift), with a clean Apple-inspired UI.

> **Status:** live Apple-style web shell with a model-agnostic form/result flow. **Cars, Loans, Bayesian
> and Wine now run on real trained weights** (Cars — gradient boosting; Loans — a stacked ensemble;
> Bayesian — a Bayesian linear regression; Wine — an extra-trees classifier); the remaining models are
> illustrative stubs for now and plug in the same way, without touching the UI. Light **and dark** theme,
> **single and batch (CSV/Excel)** prediction, and an **interactive decision-threshold slider** for
> binary classifiers (the label and precision/recall update live as you drag).

## Stack
- **Backend:** FastAPI, pydantic v2, joblib, scikit-learn (pinned `~=1.9`), pandas — layered
  (`api → services → repositories`). File uploads via `python-multipart`; `.xlsx` via `openpyxl`.
- **Frontend:** React + TypeScript (Vite, Tailwind, react-router), strict TS. Apple-style design,
  CSS-variable theming (light/dark).
- **Infra:** Docker + docker-compose, pytest.

## How it works
Each model ships as an artifact `backend/models/<name>.joblib` = `{"model": estimator, "meta": ModelInfo}`.
The API exposes the catalog; the frontend **builds the input form automatically** from each model's
feature spec and renders the result (number for regression, class + probabilities for classification).
Swapping the stub estimator for a real trained one requires **no UI changes**.

Real models are trained by `backend/training/<name>.py`, which reads `data/<name>/` and writes the
artifact (+ a `*.model_card.json` with metrics/provenance). Cars and Loans use this today; the source
research notebooks live in `notebooks/<name>/`.

```
frontend (React) → api/ (routers) → services/ (inference + file ingest) → repositories/ (model registry)
```
Inference is stateless — no DB/Redis by design (`repositories/` is the data layer for artifacts).
Beyond single prediction, `POST /predict-batch` runs a whole uploaded CSV/Excel through the model.

## Run locally
```bash
# Backend
cd backend
uv venv --python 3.12
uv pip install fastapi "uvicorn[standard]" pydantic pydantic-settings joblib "scikit-learn~=1.9.0" \
  pandas numpy python-multipart openpyxl pytest
python scripts/build_stub_models.py        # 1 demo stub artifact (uplift); the rest are real
python training/cars.py                    # train the real Cars model from data/cars/ → cars.joblib
python training/loans.py                   # train the real Loans model from data/loans/ → loans.joblib
python training/bayesian.py                # train the real Bayesian wage model → bayesian.joblib
python training/wine.py                    # train the real Wine (good ≥ 7) model → wine.joblib
python training/diamonds.py                # train the real Diamonds price model → diamonds.joblib
uv run uvicorn app.main:app --reload       # :8000

# Frontend (another terminal)
cd frontend
npm install && npm run dev                 # :5173, /api proxied to :8000
```

## Run in Docker
```bash
docker compose up --build                  # frontend :3000, backend :8000
```

## Models (catalog)
**Wine (good ≥ 7 — real weights)** · **Diamonds (price — real weights)** · **Cars (price — real weights)** ·
**Bayesian (wage — real weights)** · **Loans (credit approval — real weights)** · Uplift (uplift score).

## API
- `GET /health` — liveness.
- `GET /api/models` — catalog with per-model feature specs.
- `POST /api/models/{name}/predict` — `{"features": {...}}` → single prediction.
- `POST /api/models/{name}/predict-batch` — upload a CSV/Excel file, get a prediction per row.

## Roadmap
- Real trained models for the remaining stubs (`training/<name>.py` reproduces them from `data/`/`notebooks/`).
- Decide `*.joblib` delivery (cars/loans, gitignored): rebuild on deploy from `data/`, commit, or Git LFS.
- Optional: SHAP/feature-importance in the result card, more model types (CV/NLP).
- Done: dark theme, single + batch prediction.
