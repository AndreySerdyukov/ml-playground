# ML Playground

Interactive web app to **run trained ML models in the browser** — pick a model, enter the inputs,
get a prediction. A single super-app hosting several tabular models (wine quality, diamond & car
price, wage, credit risk, uplift), with a clean Apple-inspired UI.

> **Status:** web shell is live with an Apple-style UI and a model-agnostic form/result flow.
> Model **weights are illustrative stubs** for now — the real trained models (from the source
> notebooks) plug in next without touching the UI.

## Stack
- **Backend:** FastAPI, pydantic v2, joblib, pandas — layered (`api → services → repositories`).
- **Frontend:** React + TypeScript (Vite, Tailwind, react-router), strict TS. Apple-style design.
- **Infra:** Docker + docker-compose, pytest.

## How it works
Each model ships as an artifact `backend/models/<name>.joblib` = `{"model": estimator, "meta": ModelInfo}`.
The API exposes the catalog; the frontend **builds the input form automatically** from each model's
feature spec and renders the result (number for regression, class + probabilities for classification).
Swapping the stub estimator for a real trained one requires **no UI changes**.

```
frontend (React) → api/ (routers) → services/ (inference) → repositories/ (model registry)
```
Inference is stateless — no DB/Redis by design (`repositories/` is the data layer for artifacts).

## Run locally
```bash
# Backend
cd backend
uv venv --python 3.12 && uv pip install fastapi "uvicorn[standard]" pydantic pydantic-settings joblib scikit-learn pandas numpy pytest
python scripts/build_stub_models.py        # generate the 6 demo model artifacts
uv run uvicorn app.main:app --reload       # :8000

# Frontend (another terminal)
cd frontend
npm install && npm run dev                 # :5173, /api proxied to :8000
```

## Run in Docker
```bash
docker compose up --build                  # frontend :3000, backend :8000
```

## Models (initial catalog)
🍷 Wine (quality) · 💎 Diamonds (price) · 🚗 Cars (price) · 💵 Bayesian (wage) ·
🏦 Loans (default risk) · 📈 Uplift (uplift score).

## API
- `GET /health` — liveness.
- `GET /api/models` — catalog with per-model feature specs.
- `POST /api/models/{name}/predict` — `{"features": {...}}` → prediction.

## Roadmap
- Replace stub predictors with real trained models (`training/` reproduces them from the notebooks).
- Optional: dark mode, SHAP/feature-importance in the result card, more model types (CV/NLP).
