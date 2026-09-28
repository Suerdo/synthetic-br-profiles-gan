"""Endpoints de governança sanitizados para a interface React."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from synthetic_br_profiles_gan.api.dependencies import get_settings
from synthetic_br_profiles_gan.api.schemas.governance import GovernanceSnapshotResponse
from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.services.governance_service import GovernanceSourceConfig, build_governance_api_snapshot

router = APIRouter(tags=["governance"])


@router.get("/governance", response_model=GovernanceSnapshotResponse)
def governance(settings: ApiSettings = Depends(get_settings)):
    """Retorna o snapshot completo de governança sem expor detalhes locais sensíveis."""
    snapshot = _snapshot_or_error(settings)
    if isinstance(snapshot, JSONResponse):
        return snapshot
    return GovernanceSnapshotResponse(**snapshot)


@router.get("/governance/summary")
def governance_summary(settings: ApiSettings = Depends(get_settings)):
    """Retorna apenas o resumo operacional."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["operational"]


@router.get("/governance/quality")
def governance_quality(settings: ApiSettings = Depends(get_settings)):
    """Retorna indicadores de qualidade de dados."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["quality"]


@router.get("/governance/privacy")
def governance_privacy(settings: ApiSettings = Depends(get_settings)):
    """Retorna indicadores de diversidade, memorização e privacidade."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["privacy"]


@router.get("/governance/income")
def governance_income(settings: ApiSettings = Depends(get_settings)):
    """Retorna indicadores agregados de realismo condicional da renda."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["income"]


@router.get("/governance/executions")
def governance_executions(settings: ApiSettings = Depends(get_settings)):
    """Retorna execuções recentes sem caminhos locais."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["executions"]


@router.get("/governance/audit")
def governance_audit(settings: ApiSettings = Depends(get_settings)):
    """Retorna eventos de auditoria sanitizados."""
    snapshot = _snapshot_or_error(settings)
    return snapshot if isinstance(snapshot, JSONResponse) else snapshot["audit"]


def _snapshot_or_error(settings: ApiSettings) -> dict[str, Any] | JSONResponse:
    try:
        return build_governance_api_snapshot(
            GovernanceSourceConfig(
                artifacts_root=settings.artifacts_root,
                models_root=settings.models_root,
                audit_events_path=settings.audit_events_path,
                default_model=settings.default_model,
                approved_model_artifacts=settings.approved_model_artifacts,
            )
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "GOVERNANCE_UNAVAILABLE",
                    "message": "Não foi possível carregar as informações de governança.",
                }
            },
        )
