"""Endpoints de modelos e artefatos administrados pela aplicação."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from synthetic_br_profiles_gan.api.dependencies import get_job_manager, get_settings
from synthetic_br_profiles_gan.api.jobs import GenerationJobManager
from synthetic_br_profiles_gan.api.schemas.models import ModelArtifactResponse, ModelArtifactsResponse, ModelsResponse
from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.models.registry import SavedModelArtifact
from synthetic_br_profiles_gan.services.model_catalog import model_catalog

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
        available = catalog_entry.name == "programmatic" or bool(artifacts)
        entries.append(
            {
                "name": catalog_entry.name,
                "label": catalog_entry.label,
                "category": catalog_entry.category,
                "status": catalog_entry.status,
                "recommended": catalog_entry.recommended,
                "experimental": catalog_entry.experimental,
                "requires_training": catalog_entry.requires_training,
                "requires_saved_artifact": catalog_entry.requires_saved_artifact,
                "available": available,
                "row_limit": int(settings.row_limits[catalog_entry.name]),
                "short_description": catalog_entry.short_description,
                "detailed_description": catalog_entry.detailed_description,
                "recommended_use_cases": list(catalog_entry.recommended_use_cases),
                "benefits": list(catalog_entry.benefits),
                "limitations": list(catalog_entry.limitations),
                "recommended_artifact": None if recommended is None else _artifact_response(recommended),
                "artifact_count": len(artifacts),
                "availability_message": None if available else _missing_artifact_message(catalog_entry.name),
                "metadata": {
                    "complexity_level": catalog_entry.complexity_level,
                    "realism_profile": catalog_entry.realism_profile,
                    "capacity_notes": catalog_entry.capacity_notes,
                    "benchmark_notes": catalog_entry.benchmark_notes,
                    "requires_saved_artifact": catalog_entry.requires_saved_artifact,
                },
            }
        )
    return ModelsResponse(models=entries, default_model=settings.default_model)


@router.get("/models/{model}/artifacts", response_model=ModelArtifactsResponse)
def model_artifacts(
    model: str,
    manager: GenerationJobManager = Depends(get_job_manager),
) -> ModelArtifactsResponse:
    """Lista artefatos válidos de um modelo sem expor caminhos locais."""
    normalized_model = model.lower().replace("-", "_")
    if normalized_model not in {"ctgan", "simple_gan", "programmatic"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Modelo não encontrado.")
    artifacts = manager.list_artifacts(normalized_model)
    recommended = manager.recommended_artifact(normalized_model)
    return ModelArtifactsResponse(
        model=normalized_model,
        artifacts=[_artifact_response(artifact) for artifact in artifacts],
        recommended_artifact_id=None if recommended is None else recommended.artifact_id,
    )


def _artifact_response(artifact: SavedModelArtifact) -> ModelArtifactResponse:
    label = _artifact_label(artifact)
    return ModelArtifactResponse(
        artifact_id=artifact.artifact_id,
        model=artifact.model,
        label=label,
        created_at_utc=artifact.created_at_utc,
        train_rows=artifact.train_rows,
        seed=artifact.seed,
        status=_status_label(artifact.approval_status or artifact.purpose),
        purpose=artifact.purpose,
        recommended_for_neural_generation=artifact.recommended_for_neural_generation,
        general_platform_default=artifact.general_platform_default,
        schema_version=artifact.schema_version,
        categorical_vocabulary_version=artifact.categorical_vocabulary_version,
        income_model_version=artifact.income_model_version,
        geography_model_version=artifact.geography_model_version,
        geography_catalog_checksum=artifact.geography_catalog_checksum,
        training_required=artifact.training_required,
        model_size_bytes=artifact.model_size_bytes,
        is_legacy_vocabulary=artifact.is_legacy_vocabulary,
        is_legacy_income_model=artifact.is_legacy_income_model,
        is_legacy_geography_model=artifact.is_legacy_geography_model,
        compatibility_normalization_required=artifact.compatibility_normalization_required,
        warning=_artifact_warning(artifact),
    )


def _artifact_label(artifact: SavedModelArtifact) -> str:
    created = artifact.created_at_utc or "sem data"
    status = _status_label(artifact.approval_status or artifact.purpose)
    return f"{artifact.artifact_id} — {created} — {status}"


def _status_label(status: str) -> str:
    labels = {
        "approved": "Aprovado",
        "recommended_candidate": "Candidato recomendado",
        "candidate": "Candidato",
        "experimental": "Experimental",
        "smoke": "Smoke",
        "legacy": "Legado",
    }
    return labels.get((status or "").lower(), "Sem classificação")


def _artifact_warning(artifact: SavedModelArtifact) -> str | None:
    status = (artifact.approval_status or artifact.purpose or "").lower()
    if artifact.is_legacy_vocabulary:
        return (
            "Este artefato utiliza uma versão anterior do vocabulário. A saída será normalizada, "
            "mas poderá apresentar menor diversidade de ocupações."
        )
    if status == "smoke":
        return "Este artefato foi treinado apenas para validação técnica e não representa um modelo de produção."
    if status == "experimental":
        return "Este artefato possui finalidade experimental. Avalie suas métricas antes de uso crítico."
    if status in {"candidate", "recommended_candidate"}:
        return "Este artefato está em avaliação e ainda não deve ser tratado como padrão geral da plataforma."
    return None


def _missing_artifact_message(model: str) -> str:
    if model == "ctgan":
        return "Nenhum artefato CTGAN válido foi encontrado. Treine ou disponibilize um modelo por meio da CLI."
    if model == "simple_gan":
        return "Nenhum artefato GAN simples válido foi encontrado. Treine ou disponibilize um modelo por meio da CLI."
    return None
