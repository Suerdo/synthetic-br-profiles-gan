"""Schemas do catálogo de colunas."""

from __future__ import annotations

from synthetic_br_profiles_gan.api.schemas.common import StrictBaseModel


class ColumnEntryResponse(StrictBaseModel):
    name: str
    label: str
    description: str
    group: str
    kind: str
    generated_by: str
    dependencies: list[str]
    sensitive_like: bool
    default_selected: bool


class ColumnPresetResponse(StrictBaseModel):
    name: str
    columns: list[str]


class ColumnsResponse(StrictBaseModel):
    final_columns: list[str]
    groups: list[str]
    columns: list[ColumnEntryResponse]
    presets: list[ColumnPresetResponse]
    default_preset: str
