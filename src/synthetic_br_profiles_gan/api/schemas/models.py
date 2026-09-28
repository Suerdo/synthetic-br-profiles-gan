"""Schemas do catálogo de modelos e artefatos."""

from __future__ import annotations

from typing import Any

from synthetic_br_profiles_gan.api.schemas.common import StrictBaseModel


class ModelArtifactResponse(StrictBaseModel):
    artifact_id: str
    model: str
    label: str
    created_at_utc: str | None
    train_rows: int | None
    seed: int | None
    status: str
    purpose: str
    recommended_for_neural_generation: bool
    general_platform_default: bool
    schema_version: int
    categorical_vocabulary_version: int
    income_model_version: int
    geography_model_version: int
    geography_catalog_checksum: str | None
    training_required: bool
    model_size_bytes: int | None
    is_legacy_vocabulary: bool
    is_legacy_income_model: bool
    is_legacy_geography_model: bool
    compatibility_normalization_required: bool
    warning: str | None


class ModelEntryResponse(StrictBaseModel):
    name: str
    label: str
    category: str
    status: str
    recommended: bool
    experimental: bool
    requires_training: bool
    requires_saved_artifact: bool
    available: bool
    row_limit: int
    short_description: str
    detailed_description: str
    recommended_use_cases: list[str]
    benefits: list[str]
    limitations: list[str]
    recommended_artifact: ModelArtifactResponse | None
    artifact_count: int
    availability_message: str | None
    metadata: dict[str, Any]


class ModelsResponse(StrictBaseModel):
    models: list[ModelEntryResponse]
    default_model: str


class ModelArtifactsResponse(StrictBaseModel):
    model: str
    artifacts: list[ModelArtifactResponse]
    recommended_artifact_id: str | None
