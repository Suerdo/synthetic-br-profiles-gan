"""Schemas públicos da governança consumida pelo frontend React."""

from __future__ import annotations

from typing import Any

from synthetic_br_profiles_gan.api.schemas.common import StrictBaseModel


class GovernanceSectionResponse(StrictBaseModel):
    """Seção genérica de indicadores agregados."""

    indicators: list[dict[str, Any]]
    status: str


class GovernancePrivacyResponse(StrictBaseModel):
    """Indicadores de privacidade, diversidade e memorização."""

    indicators: list[dict[str, Any]]
    diversity_memorization: list[dict[str, Any]]
    status: str


class GovernanceOperationalResponse(StrictBaseModel):
    """Resumo operacional sanitizado."""

    metrics: list[dict[str, Any]]
    summary: dict[str, Any]


class GovernanceSnapshotResponse(StrictBaseModel):
    """Snapshot público de governança sem caminhos locais nem dados sensíveis."""

    operational: GovernanceOperationalResponse
    recommended_model: dict[str, Any] | None
    quality: GovernanceSectionResponse
    privacy: GovernancePrivacyResponse
    income: GovernanceSectionResponse
    executions: list[dict[str, Any]]
    audit: list[dict[str, Any]]
    glossary: list[dict[str, str]]
