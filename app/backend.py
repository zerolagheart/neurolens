"""Inference backend for the UI.

By default the app uses :class:`api.service.ModelService` in-process (same code and schemas as
the API). When ``NEUROLENS_API_URL`` is set (e.g. in Docker Compose), it calls the FastAPI
service over HTTP instead and parses the same Pydantic response models.
"""

from __future__ import annotations

from typing import Protocol

import httpx
import streamlit as st

from api.schemas import ExplainResponse, HealthResponse, ModelsResponse
from api.service import ModelService
from api.settings import Settings
from ui import API_URL, RUNS_DIR


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    mode: str

    def health(self) -> HealthResponse: ...
    def models(self) -> ModelsResponse: ...
    def explain(self, data: bytes, filename: str, model: str | None) -> ExplainResponse: ...


class LocalBackend:
    mode = "in-process"

    def __init__(self) -> None:
        settings = Settings.from_env()
        settings.runs_dir = RUNS_DIR
        self.service = ModelService(settings)
        self.service.load()

    def health(self) -> HealthResponse:
        return self.service.health()

    def models(self) -> ModelsResponse:
        return self.service.models()

    def explain(self, data: bytes, filename: str, model: str | None) -> ExplainResponse:
        from neurolens.inference.image_io import InvalidImageError

        try:
            return self.service.explain(data, model)
        except (InvalidImageError, KeyError, RuntimeError) as exc:
            raise BackendError(exc.args[0] if exc.args else str(exc)) from exc


class HttpBackend:
    mode = "api"

    def __init__(self, base_url: str) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=60.0)

    def _get(self, path: str) -> dict:
        try:
            response = self.client.get(path)
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach the API at {self.client.base_url}: {exc}") from exc
        response.raise_for_status()
        return response.json()

    def health(self) -> HealthResponse:
        return HealthResponse(**self._get("/health"))

    def models(self) -> ModelsResponse:
        return ModelsResponse(**self._get("/models"))

    def explain(self, data: bytes, filename: str, model: str | None) -> ExplainResponse:
        try:
            response = self.client.post(
                "/explain",
                files={"file": (filename, data)},
                params={"model": model} if model else None,
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach the API: {exc}") from exc
        if response.status_code != 200:
            detail = (
                response.json().get("detail", response.text)
                if response.content
                else response.reason_phrase
            )
            raise BackendError(f"API error {response.status_code}: {detail}")
        return ExplainResponse(**response.json())


@st.cache_resource(show_spinner="Loading trained models…")
def get_backend() -> Backend:
    return HttpBackend(API_URL) if API_URL else LocalBackend()
