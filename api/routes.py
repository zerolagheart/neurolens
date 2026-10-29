"""HTTP endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from api.schemas import (
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    ModelsResponse,
    PredictionResponse,
)
from api.service import ModelNotReadyError, ModelService, UnknownModelError
from neurolens.inference.image_io import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
)
from neurolens.utils import get_logger

log = get_logger("api.routes")
router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Corrupt or undecodable image"},
    404: {"model": ErrorResponse, "description": "Unknown model name"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported image format"},
    503: {"model": ErrorResponse, "description": "No trained model is loaded"},
}
ModelParam = Annotated[
    str | None, Query(description="Run name from GET /models; defaults to the service default")
]


def _service(request: Request) -> ModelService:
    return request.app.state.service


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    # Read at most limit + 1 bytes so oversized uploads are rejected without buffering them all.
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds the {limit / 1024 / 1024:.1f} MB upload limit.")
    return data


async def _run(service: ModelService, endpoint: str, func, data: bytes, model: str | None):
    started = time.perf_counter()
    ok = False
    try:
        # Inference is CPU/GPU-bound; run it in the thread pool to keep the event loop free.
        result = await run_in_threadpool(func, data, model)
        ok = True
        return result
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except UnknownModelError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, exc.args[0]) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    finally:
        elapsed = time.perf_counter() - started
        service.record(endpoint, elapsed, ok)
        # Never log image content or the client's filename.
        log.info(
            "%s model=%s bytes=%d ok=%s latency_ms=%.1f",
            endpoint,
            model or "default",
            len(data),
            ok,
            elapsed * 1000,
        )


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    """Liveness plus model readiness. ``ready`` is false until a trained checkpoint is loaded."""
    return _service(request).health()


@router.get("/models", response_model=ModelsResponse, tags=["models"])
def models(request: Request) -> ModelsResponse:
    """Supported architectures and the trained checkpoints discovered on the server."""
    return _service(request).models()


@router.post(
    "/predict", response_model=PredictionResponse, responses=ERROR_RESPONSES, tags=["inference"]
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="MRI image (JPEG, PNG, BMP or TIFF)")],
    model: ModelParam = None,
) -> PredictionResponse:
    """Classify one image. Research use only - not a medical diagnosis."""
    service = _service(request)
    data = await _read_upload(file, service.settings.max_upload_bytes)
    return await _run(service, "predict", service.predict, data, model)


@router.post(
    "/explain", response_model=ExplainResponse, responses=ERROR_RESPONSES, tags=["inference"]
)
async def explain(
    request: Request,
    file: Annotated[UploadFile, File(description="MRI image (JPEG, PNG, BMP or TIFF)")],
    model: ModelParam = None,
) -> ExplainResponse:
    """Prediction plus Grad-CAM heatmap/overlay as base64-encoded PNGs."""
    service = _service(request)
    data = await _read_upload(file, service.settings.max_upload_bytes)
    return await _run(service, "explain", service.explain, data, model)


@router.get("/metrics", response_model=MetricsResponse, tags=["models"])
def metrics(request: Request) -> MetricsResponse:
    """Saved offline evaluation results, plus live statistics of this process (kept separate)."""
    return _service(request).metrics()
