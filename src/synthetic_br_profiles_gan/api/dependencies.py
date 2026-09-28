"""Dependências compartilhadas dos endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import Header, HTTPException, Request, status

from synthetic_br_profiles_gan.api.jobs import GenerationJobManager
from synthetic_br_profiles_gan.api.settings import ApiSettings


def get_settings(request: Request) -> ApiSettings:
    """Retorna a configuração associada à aplicação FastAPI."""
    return request.app.state.settings


def get_job_manager(request: Request) -> GenerationJobManager:
    """Retorna o gerenciador de jobs associado à aplicação."""
    return request.app.state.job_manager


def require_ui_session_id(x_ui_session_id: str | None = Header(default=None)) -> str:
    """Valida o cabeçalho efêmero de sessão da interface."""
    if not x_ui_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Informe o cabeçalho X-UI-Session-ID para isolar a geração desta sessão.",
        )
    try:
        return str(UUID(str(x_ui_session_id)))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O cabeçalho X-UI-Session-ID deve conter um UUID válido.",
        ) from exc
