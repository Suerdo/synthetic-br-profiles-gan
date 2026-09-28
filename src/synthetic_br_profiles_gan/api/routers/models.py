"""Endpoints de modelos e artefatos administrados pela aplicação."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from synthetic_br_profiles_gan.api.dependencies import get_job_manager, get_settings
from synthetic_br_profiles_gan.api.jobs import GenerationJobManager
from synthetic_br_profiles_gan.api.schemas.models import (
    ModelArtifactResponse,
    ModelArtifactsResponse,
    ModelEntryResponse,
    ModelsResponse,
    RecommendedArtifactResponse,
)
from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.models.registry import SavedModelArtifact
from synthetic_br_profiles_gan.services.governance_service import artifact_quality_summary
from synthetic_br_profiles_gan.services.model_catalog import ModelCatalogEntry, model_catalog, model_catalog_by_name

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def models(
    settings: ApiSettings = Depends(get_settings),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> ModelsResponse:
    """Lista os três tipos de modelo e sua disponibilidade para a interface."""
    entries = []
    for catalog_entry in model_catalog():
        artifacts = manager.list_artifacts(catalog_entry.name)
        recommended = manager.recommended_artifact(catalog_entry.name)
        entries.append(_model_entry_response(catalog_entry, artifacts, recommended, settings))
    return ModelsResponse(models=entries, default_model=settings.default_model)


@router.get("/models/{model}", response_model=ModelEntryResponse)
def model_detail(
    model: str,
    settings: ApiSettings = Depends(get_settings),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> ModelEntryResponse:
    """Retorna o contrato completo de um modelo específico."""
    normalized_model = _normalize_model_or_404(model)
    catalog_entry = model_catalog_by_name()[normalized_model]
    artifacts = manager.list_artifacts(normalized_model)
    recommended = manager.recommended_artifact(normalized_model)
    return _model_entry_response(catalog_entry, artifacts, recommended, settings)


@router.get("/models/{model}/artifacts", response_model=ModelArtifactsResponse)
def model_artifacts(
    model: str,
    manager: GenerationJobManager = Depends(get_job_manager),
) -> ModelArtifactsResponse:
    """Lista artefatos válidos de um modelo sem expor caminhos locais."""
    normalized_model = _normalize_model_or_404(model)
    artifacts = manager.list_artifacts(normalized_model)
    recommended = manager.recommended_artifact(normalized_model)
    return ModelArtifactsResponse(
        model=normalized_model,
        artifacts=[_artifact_response(artifact) for artifact in artifacts],
        recommended_artifact_id=None if recommended is None else recommended.artifact_id,
    )


@router.get("/models/{model}/recommended", response_model=RecommendedArtifactResponse)
def recommended_artifact(
    model: str,
    manager: GenerationJobManager = Depends(get_job_manager),
) -> RecommendedArtifactResponse:
    """Retorna o artefato recomendado do modelo, quando existir."""
    normalized_model = _normalize_model_or_404(model)
    if normalized_model == "programmatic":
        return RecommendedArtifactResponse(
            model=normalized_model,
            artifact=None,
            message="O modelo programático não exige artefato salvo.",
        )
    artifact = manager.recommended_artifact(normalized_model)
    return RecommendedArtifactResponse(
        model=normalized_model,
        artifact=None if artifact is None else _artifact_response(artifact),
        message=None if artifact is not None else _missing_artifact_message(normalized_model),
    )


def _model_entry_response(
    catalog_entry: ModelCatalogEntry,
    artifacts: list[SavedModelArtifact],
    recommended: SavedModelArtifact | None,
    settings: ApiSettings,
) -> ModelEntryResponse:
    available = catalog_entry.name == "programmatic" or bool(artifacts)
    return ModelEntryResponse(
        id=catalog_entry.name,
        name=catalog_entry.name,
        title=catalog_entry.label,
        label=catalog_entry.label,
        category=catalog_entry.category,
        status=catalog_entry.status,
        recommended=catalog_entry.recommended,
        experimental=catalog_entry.experimental,
        requires_training=catalog_entry.requires_training,
        requires_saved_artifact=catalog_entry.requires_saved_artifact,
        available=available,
        row_limit=int(settings.row_limits[catalog_entry.name]),
        short_description=catalog_entry.short_description,
        detailed_description=catalog_entry.detailed_description,
        summary=catalog_entry.simple_summary,
        simple_summary=catalog_entry.simple_summary,
        technical_summary=catalog_entry.technical_summary,
        recommended_for=_recommended_for_text(catalog_entry),
        recommended_use_cases=list(catalog_entry.recommended_use_cases),
        benefits=list(catalog_entry.benefits),
        limitations=list(catalog_entry.limitations),
        governance_notes=list(catalog_entry.privacy_considerations + catalog_entry.compliance_considerations),
        recommended_artifact=None if recommended is None else _artifact_response(recommended),
        artifact_count=len(artifacts),
        availability_message=None if available else _missing_artifact_message(catalog_entry.name),
        metadata={
            "complexity_level": catalog_entry.complexity_level,
            "realism_profile": catalog_entry.realism_profile,
            "capacity_notes": catalog_entry.capacity_notes,
            "benchmark_notes": catalog_entry.benchmark_notes,
            "requires_saved_artifact": catalog_entry.requires_saved_artifact,
        },
    )


def _artifact_response(artifact: SavedModelArtifact) -> ModelArtifactResponse:
    quality = artifact_quality_summary(artifact)
    normalized_status = (artifact.approval_status or artifact.purpose or "").lower()
    return ModelArtifactResponse(
        artifact_id=artifact.artifact_id,
        model=artifact.model,
        label=_artifact_label(artifact),
        created_at_utc=artifact.created_at_utc,
        created_at=artifact.created_at_utc,
        train_rows=artifact.train_rows,
        seed=artifact.seed,
        status=_status_label(artifact.approval_status or artifact.purpose),
        purpose=artifact.purpose,
        approved=normalized_status == "approved" or artifact.purpose == "approved",
        recommended=artifact.recommended_for_neural_generation,
        recommended_for_neural_generation=artifact.recommended_for_neural_generation,
        general_platform_default=artifact.general_platform_default,
        schema_version=artifact.schema_version,
        categorical_vocabulary_version=artifact.categorical_vocabulary_version,
        vocabulary_version=artifact.categorical_vocabulary_version,
        income_model_version=artifact.income_model_version,
        geography_model_version=artifact.geography_model_version,
        geography_catalog_checksum=artifact.geography_catalog_checksum,
        training_required=artifact.training_required,
        model_size_bytes=artifact.model_size_bytes,
        epochs=_int_or_none(quality.get("epochs")),
        library=_string_or_none(quality.get("library")),
        quality_status=_string_or_none(quality.get("quality_status")),
        duplicate_base_row_rate=quality.get("duplicate_base_row_rate"),
        exact_train_match_rate=quality.get("exact_train_match_rate"),
        conditional_income_status=_string_or_none(quality.get("conditional_income_status")),
        compatibility=_compatibility_label(artifact),
        is_legacy_vocabulary=artifact.is_legacy_vocabulary,
        is_legacy_income_model=artifact.is_legacy_income_model,
        is_legacy_geography_model=artifact.is_legacy_geography_model,
        compatibility_normalization_required=artifact.compatibility_normalization_required,
        warning=_artifact_warning(artifact),
    )


def _normalize_model_or_404(model: str) -> str:
    normalized_model = model.lower().replace("-", "_")
    if normalized_model not in model_catalog_by_name():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Modelo não encontrado.")
    return normalized_model


def _recommended_for_text(entry: ModelCatalogEntry) -> str:
    if not entry.recommended_use_cases:
        return "Indicado para: Não avaliado."
    use_cases = [str(item) for item in entry.recommended_use_cases]
    first = use_cases[0].capitalize()
    if len(use_cases) == 1:
        text = first
    else:
        text = ", ".join([first, *use_cases[1:-1]]) + f" e {use_cases[-1]}"
    return f"Indicado para: {text}."


def _artifact_label(artifact: SavedModelArtifact) -> str:
    created = artifact.created_at_utc or "sem data"
    status_label = _status_label(artifact.approval_status or artifact.purpose)
    return f"{artifact.artifact_id} — {created} — {status_label}"


def _status_label(value: str) -> str:
    labels = {
        "approved": "Aprovado",
        "recommended_candidate": "Candidato recomendado",
        "candidate": "Candidato",
        "experimental": "Experimental",
        "smoke": "Smoke",
        "legacy": "Legado",
    }
    return labels.get((value or "").lower(), "Sem classificação")


def _artifact_warning(artifact: SavedModelArtifact) -> str | None:
    status_value = (artifact.approval_status or artifact.purpose or "").lower()
    if artifact.is_legacy_vocabulary:
        return (
            "Este artefato utiliza uma versão anterior do vocabulário. A saída será normalizada, "
            "mas poderá apresentar menor diversidade de ocupações."
        )
    if status_value == "smoke":
        return "Este artefato foi treinado apenas para validação técnica e não representa um modelo de produção."
    if status_value == "experimental":
        return "Este artefato possui finalidade experimental. Avalie suas métricas antes de uso crítico."
    if status_value in {"candidate", "recommended_candidate"}:
        return "Este artefato está em avaliação e ainda não deve ser tratado como padrão geral da plataforma."
    return None


def _compatibility_label(artifact: SavedModelArtifact) -> str:
    if (
        artifact.compatibility_normalization_required
        or artifact.is_legacy_vocabulary
        or artifact.is_legacy_income_model
        or artifact.is_legacy_geography_model
    ):
        return "compatibilidade legada com normalização"
    return "compatível"


def _missing_artifact_message(model: str) -> str | None:
    if model == "ctgan":
        return "Nenhum artefato CTGAN válido foi encontrado. Treine ou disponibilize um modelo por meio da CLI."
    if model == "simple_gan":
        return "Nenhum artefato GAN simples válido foi encontrado. Treine ou disponibilize um modelo por meio da CLI."
    return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
