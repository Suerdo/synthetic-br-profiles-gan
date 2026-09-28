"""Schemas das solicitações e respostas de geração."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from synthetic_br_profiles_gan.api.schemas.common import StrictBaseModel


GenerationModel = Literal["programmatic", "ctgan", "simple_gan"]
OutputFormat = Literal["csv", "json", "parquet"]
GenerationStatus = Literal["queued", "running", "completed", "failed"]


class GenerationCreateRequest(StrictBaseModel):
    model: GenerationModel
    artifact_id: str | None = None
    num_rows: int = Field(..., ge=1)
    output_format: OutputFormat = "csv"
    seed: int = Field(41, ge=0)
    selected_columns: list[str] | None = None
    column_preset: str | None = None


class GenerationCreateResponse(StrictBaseModel):
    generation_id: str
    status: GenerationStatus
    status_url: str


class GenerationStatusResponse(StrictBaseModel):
    generation_id: str
    status: GenerationStatus
    model: str
    artifact_id: str | None
    num_rows: int
    output_format: str
    seed: int
    submitted_at_utc: str
    started_at_utc: str | None
    completed_at_utc: str | None
    duration_seconds: float | None
    exported_columns: list[str]
    internal_columns: list[str]
    validation: dict[str, Any] | None
    dataset_download_url: str | None
    manifest_download_url: str | None
    preview_url: str | None
    error: dict[str, str] | None


class GenerationPreviewResponse(StrictBaseModel):
    generation_id: str
    rows: list[dict[str, Any]]
    columns: list[str]
    preview_rows: int
    total_rows: int


class GenerationManifestResponse(StrictBaseModel):
    generation_id: str
    manifest: dict[str, Any]
