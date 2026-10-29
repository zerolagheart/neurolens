"""FastAPI application factory.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.service import ModelService
from api.settings import Settings
from neurolens import DISCLAIMER, __version__
from neurolens.utils import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = ModelService(settings)
        service.load()  # models are loaded once here, never per request
        app.state.service = service
        yield

    app = FastAPI(
        title="NeuroLens API",
        version=__version__,
        description=(
            "Explainable brain MRI image classification for research and education.\n\n"
            f"**{DISCLAIMER}**"
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
