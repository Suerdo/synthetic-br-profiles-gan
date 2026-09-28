"""Endpoints do catálogo estruturado de colunas."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from synthetic_br_profiles_gan.api.dependencies import get_settings
from synthetic_br_profiles_gan.api.schemas.columns import ColumnsResponse
from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.column_catalog import COLUMN_CATALOG, COLUMN_PRESETS
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS

router = APIRouter(tags=["columns"])


@router.get("/columns", response_model=ColumnsResponse)
def columns(settings: ApiSettings = Depends(get_settings)) -> ColumnsResponse:
    """Retorna colunas finais, grupos e presets reutilizáveis pelo frontend."""
    groups: list[str] = []
    for entry in COLUMN_CATALOG:
        if entry.group not in groups:
            groups.append(entry.group)
    return ColumnsResponse(
        final_columns=list(FINAL_COLUMNS),
        groups=groups,
        columns=[
            {
                "name": entry.name,
                "label": entry.label,
                "description": entry.description,
                "group": entry.group,
                "kind": entry.kind,
                "generated_by": entry.generated_by,
                "dependencies": list(entry.dependencies),
                "sensitive_like": bool(entry.sensitive_like),
                "default_selected": bool(entry.default_selected),
            }
            for entry in COLUMN_CATALOG
        ],
        presets=[{"name": name, "columns": list(columns)} for name, columns in COLUMN_PRESETS.items()],
        default_preset=settings.default_preset,
    )
