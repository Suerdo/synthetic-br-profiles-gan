"""Aplicação FastAPI da interface React."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from synthetic_br_profiles_gan.api.errors import register_exception_handlers
from synthetic_br_profiles_gan.api.jobs import GenerationJobManager
from synthetic_br_profiles_gan.api.routers import columns, generations, governance, health, models
from synthetic_br_profiles_gan.api.settings import ApiSettings, load_api_settings


def create_app(settings: ApiSettings | None = None, job_manager: GenerationJobManager | None = None) -> FastAPI:
    """Cria a aplicação HTTP com dependências injetáveis para testes."""
    resolved_settings = settings or load_api_settings()
    app = FastAPI(
        title=resolved_settings.title,
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = resolved_settings
    app.state.job_manager = job_manager or GenerationJobManager(resolved_settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-UI-Session-ID"],
    )
    app.include_router(health.router, prefix="/api")
    app.include_router(columns.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(governance.router, prefix="/api")
    app.include_router(generations.router, prefix="/api")
    register_exception_handlers(app)
    return app


app = create_app()
