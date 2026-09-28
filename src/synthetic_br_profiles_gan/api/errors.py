"""Tratamento de erros esperados da API."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from synthetic_br_profiles_gan.exceptions import (
    ConfigurationError,
    ModelBackendUnavailable,
    ModelSerializationError,
    PipelineError,
    StructuralValidationError,
    SyntheticModelError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Registra conversões de exceções de domínio para respostas sanitizadas."""

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, "configuration_error", str(exc))

    @app.exception_handler(ModelSerializationError)
    async def model_serialization_error_handler(request: Request, exc: ModelSerializationError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "model_artifact_error", str(exc))

    @app.exception_handler(ModelBackendUnavailable)
    async def model_backend_error_handler(request: Request, exc: ModelBackendUnavailable) -> JSONResponse:
        return _error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "model_backend_unavailable", str(exc))

    @app.exception_handler(StructuralValidationError)
    async def structural_error_handler(request: Request, exc: StructuralValidationError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "structural_validation_error", str(exc))

    @app.exception_handler(SyntheticModelError)
    async def synthetic_model_error_handler(request: Request, exc: SyntheticModelError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "synthetic_model_error", str(exc))

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, "pipeline_error", str(exc))


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})
