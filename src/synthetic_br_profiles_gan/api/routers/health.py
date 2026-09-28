"""Endpoint de saúde operacional da API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from synthetic_br_profiles_gan.api.dependencies import get_settings
from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.models.registry import list_saved_model_artifacts

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: ApiSettings = Depends(get_settings)) -> dict[str, object]:
    """Retorna um resumo sem caminhos locais sensíveis."""
    ctgan_available = bool(list_saved_model_artifacts(settings.models_root, model="ctgan"))
    simple_gan_available = bool(list_saved_model_artifacts(settings.models_root, model="simple_gan"))
    return {
        "status": "ok",
        "service": "synthetic-br-profiles-gan-api",
        "api_version": "0.1.0",
        "defaults": {
            "model": settings.default_model,
            "rows": settings.default_rows,
            "format": settings.default_format,
            "seed": settings.default_seed,
            "preset": settings.default_preset,
        },
        "models": {
            "programmatic": {"available": True, "requires_artifact": False},
            "ctgan": {"available": ctgan_available, "requires_artifact": True},
            "simple_gan": {"available": simple_gan_available, "requires_artifact": True},
        },
    }
