"""Compatibilidade temporária para o serviço de governança da UI."""

from synthetic_br_profiles_gan.services.governance_service import (
    GovernanceSnapshot,
    GovernanceSourceConfig,
    artifact_public_summary,
    artifact_quality_summary,
    build_governance_api_snapshot,
    build_governance_snapshot,
    default_generation_model,
    governance_glossary,
    is_approved_vocabulary_v2_artifact,
    model_version_rows,
    recommended_neural_model_public,
)

__all__ = [
    "GovernanceSnapshot",
    "GovernanceSourceConfig",
    "artifact_public_summary",
    "artifact_quality_summary",
    "build_governance_api_snapshot",
    "build_governance_snapshot",
    "default_generation_model",
    "governance_glossary",
    "is_approved_vocabulary_v2_artifact",
    "model_version_rows",
    "recommended_neural_model_public",
]
