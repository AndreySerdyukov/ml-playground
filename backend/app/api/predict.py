"""Inference router: model listing and prediction.

The router is thin: it only pulls the service out of the application state, catches domain
exceptions and translates them into HTTP responses. There is no business logic here.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas.predict import (
    BatchPredictResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from app.services.file_ingest import parse_table
from app.services.inference import (
    InferenceService,
    InvalidFeaturesError,
    ModelNotFoundError,
    PredictionError,
)

router = APIRouter(prefix="/api", tags=["inference"])

# Upper bound on the uploaded file size (protection against giant uploads).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _service(request: Request) -> InferenceService:
    """Fetch the singleton service from app.state (placed there in main.create_app)."""
    service: InferenceService = request.app.state.inference_service
    return service


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request) -> list[ModelInfo]:
    """List of available models with feature descriptions (for building the form)."""
    return _service(request).list_models()


@router.post("/models/{model_name}/predict", response_model=PredictResponse)
def predict(model_name: str, payload: PredictRequest, request: Request) -> PredictResponse:
    """Prediction with the selected model from the supplied features."""
    try:
        return _service(request).predict(model_name, payload.features)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}") from exc
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=f"Could not score inputs: {exc}") from exc


@router.post("/models/{model_name}/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(
    model_name: str, request: Request, file: Annotated[UploadFile, File()]
) -> BatchPredictResponse:
    """Batch predict: a CSV/Excel is uploaded, the response is a prediction for each row."""
    # Reject oversized uploads early (from the declared size) before buffering the whole body.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (maximum 10 MB)")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (maximum 10 MB)")
    try:
        records = parse_table(file.filename or "", raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return _service(request).predict_batch(model_name, records)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}") from exc
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=f"Could not score inputs: {exc}") from exc
