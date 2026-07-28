"""Serviço de governança para indicadores e evidências da interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.domain.occupations import OCCUPATION_CATALOG
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, DATA_LOCALE, UNICODE_NORMALIZATION
from synthetic_br_profiles_gan.models.registry import SavedModelArtifact, list_saved_model_artifacts
from synthetic_br_profiles_gan.ui.services.audit_service import read_audit_events
from synthetic_br_profiles_gan.ui.services.execution_history import HistoryRecord, history_summary, load_history
from synthetic_br_profiles_gan.ui.ui_config import UIConfig


@dataclass(frozen=True)
class GovernanceSnapshot:
    """Conjunto de indicadores exibidos na área de governança."""

    overview: dict[str, Any]
    quality_indicators: list[dict[str, Any]]
    privacy_indicators: list[dict[str, Any]]
    risk_indicators: list[dict[str, Any]]
    pipeline_status: dict[str, Any]
    model_versions: list[dict[str, Any]]
    history: list[HistoryRecord]
    audit_events: list[dict[str, Any]]


def build_governance_snapshot(config: UIConfig) -> GovernanceSnapshot:
    """Monta indicadores de governança apenas com evidências locais disponíveis."""
    history = load_history(config.artifacts_root)
    summary = history_summary(history)
    artifacts = list_saved_model_artifacts(config.models_root)
    approved_generation_artifacts = [
        artifact for artifact in artifacts if artifact.model in {"ctgan", "simple_gan"} and is_approved_vocabulary_v2_artifact(artifact, config)
    ]
    latest_approved = _latest_artifact(approved_generation_artifacts)
    overview = {
        "default_model": default_generation_model(config, artifacts),
        "available_models": _available_model_labels(approved_generation_artifacts),
        "vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "data_locale": DATA_LOCALE,
        "unicode_normalization": UNICODE_NORMALIZATION,
        "total_runs": summary["total_records"],
        "latest_run": summary["latest_identifier"] or "Sem execução registrada",
        "latest_status": summary["latest_status"] or "Não avaliado",
        "occupation_count": len(OCCUPATION_CATALOG),
        "pipeline_status": "Operacional" if summary["total_records"] > 0 else "Sem execução registrada",
        "latest_approved_model_version": (
            latest_approved.artifact_id if latest_approved is not None else "Não disponível"
        ),
    }
    return GovernanceSnapshot(
        overview=overview,
        quality_indicators=_quality_indicators(history),
        privacy_indicators=_privacy_indicators(history),
        risk_indicators=_risk_indicators(history),
        pipeline_status=_pipeline_status(summary),
        model_versions=model_version_rows(artifacts, config),
        history=history,
        audit_events=read_audit_events(config.audit_events_path, limit=200),
    )


def is_approved_vocabulary_v2_artifact(artifact: SavedModelArtifact, config: UIConfig) -> bool:
    """Indica se um artefato neural pode ser usado diretamente na geração pela interface."""
    if artifact.model not in {"ctgan", "simple_gan"}:
        return False
    configured = artifact.artifact_id in set(config.approved_model_artifacts.get(artifact.model, ()))
    approved = artifact.approval_status == "approved" or artifact.purpose == "approved" or configured
    return (
        approved
        and artifact.categorical_vocabulary_version >= CATEGORICAL_VOCABULARY_VERSION
        and artifact.data_locale == DATA_LOCALE
        and artifact.unicode_normalization == UNICODE_NORMALIZATION
    )


def default_generation_model(config: UIConfig, artifacts: list[SavedModelArtifact] | None = None) -> str:
    """Escolhe o modelo padrão operacional da interface."""
    if config.default_model == "programmatic":
        return "programmatic"
    materialized = artifacts if artifacts is not None else list_saved_model_artifacts(config.models_root)
    if config.default_model == "ctgan" and any(
        artifact.model == "ctgan" and is_approved_vocabulary_v2_artifact(artifact, config)
        for artifact in materialized
    ):
        return "ctgan"
    return "programmatic"


def model_version_rows(artifacts: list[SavedModelArtifact], config: UIConfig) -> list[dict[str, Any]]:
    """Converte artefatos em linhas de histórico de modelos."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        generation_ready = is_approved_vocabulary_v2_artifact(artifact, config)
        rows.append(
            {
                "modelo": artifact.model,
                "artefato": artifact.artifact_id,
                "criado_em_utc": artifact.created_at_utc or "Não disponível",
                "treino": artifact.train_rows if artifact.train_rows is not None else "Não disponível",
                "seed": artifact.seed if artifact.seed is not None else "Não disponível",
                "schema": artifact.schema_version,
                "vocabulário": artifact.categorical_vocabulary_version,
                "localidade": artifact.data_locale or "Não disponível",
                "normalização": artifact.unicode_normalization or "Não disponível",
                "propósito": artifact.purpose,
                "status": artifact.approval_status,
                "legado": "sim" if artifact.is_legacy_vocabulary else "não",
                "disponível_para_geração": "sim" if generation_ready else "não",
            }
        )
    return rows


def _available_model_labels(approved_generation_artifacts: list[SavedModelArtifact]) -> str:
    available = {"programmatic"}
    available.update(artifact.model for artifact in approved_generation_artifacts)
    labels = {
        "programmatic": "Programático",
        "ctgan": "CTGAN",
        "simple_gan": "GAN simples",
    }
    return ", ".join(labels[model] for model in ("programmatic", "ctgan", "simple_gan") if model in available)


def _latest_artifact(artifacts: list[SavedModelArtifact]) -> SavedModelArtifact | None:
    if not artifacts:
        return None
    return sorted(artifacts, key=lambda item: item.created_at_utc or "", reverse=True)[0]


def _quality_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    latest_pipeline = next((record for record in history if record.kind in {"pipeline_run", "ui_generation"}), None)
    if latest_pipeline is None:
        return [_empty_indicator("Qualidade estrutural", "validation.is_valid")]
    validation = latest_pipeline.manifest.get("validation") or {}
    if not isinstance(validation, dict):
        return [_empty_indicator("Qualidade estrutural", "validation.is_valid")]
    is_valid = validation.get("is_valid")
    reason_counts = validation.get("reason_counts") or {}
    return [
        {
            "indicador": "Validação estrutural",
            "risco": "Baixo" if is_valid else "Elevado",
            "métrica": "validation.is_valid",
            "valor": bool(is_valid),
            "limite": "True",
            "fonte": latest_pipeline.identifier,
            "interpretação": "O schema completo foi validado antes da exportação." if is_valid else "Há falhas estruturais registradas.",
            "data": latest_pipeline.created_at_utc or "Não disponível",
        },
        {
            "indicador": "Erros estruturais",
            "risco": "Baixo" if not reason_counts else "Moderado",
            "métrica": "validation.reason_counts",
            "valor": sum(int(value) for value in reason_counts.values()) if isinstance(reason_counts, dict) else "Não avaliado",
            "limite": 0,
            "fonte": latest_pipeline.identifier,
            "interpretação": "Contagem agregada de violações estruturais do relatório mais recente.",
            "data": latest_pipeline.created_at_utc or "Não disponível",
        },
    ]


def _privacy_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    for record in history:
        manifest = record.manifest
        evaluation = manifest.get("evaluation") if isinstance(manifest, dict) else None
        generation_accounting = manifest.get("generation_accounting") if isinstance(manifest, dict) else None
        if isinstance(evaluation, dict) or isinstance(generation_accounting, dict):
            duplicate_rate = _find_metric(manifest, "duplicate_row_rate")
            train_match = _find_metric(manifest, "exact_train_match_rate")
            return [
                _privacy_indicator("Duplicidade de linhas", "duplicate_row_rate", duplicate_rate, record),
                _privacy_indicator("Correspondência exata com treino", "exact_train_match_rate", train_match, record),
            ]
    return [
        _empty_indicator("Duplicidade de linhas", "duplicate_row_rate"),
        _empty_indicator("Correspondência exata com treino", "exact_train_match_rate"),
    ]


def _risk_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    latest = history[0] if history else None
    if latest is None:
        return [
            {
                "indicador": "Risco geral operacional",
                "risco": "Não avaliado",
                "métrica": "manifest.status",
                "valor": "Sem execução registrada",
                "limite": "Não aplicável",
                "fonte": "Não disponível",
                "interpretação": "Ainda não há manifesto local para avaliar.",
                "data": "Não disponível",
            }
        ]
    status = latest.status or "Não avaliado"
    risk = "Baixo" if status in {"approved", "completed"} else "Moderado" if "quarantine" in status or status == "Não avaliado" else "Elevado"
    return [
        {
            "indicador": "Risco geral operacional",
            "risco": risk,
            "métrica": "manifest.status",
            "valor": status,
            "limite": "approved/completed para baixo risco técnico",
            "fonte": latest.identifier,
            "interpretação": "Leitura conservadora do status mais recente; não é certificação de conformidade.",
            "data": latest.created_at_utc or "Não disponível",
        }
    ]


def _pipeline_status(summary: dict[str, Any]) -> dict[str, Any]:
    if summary["total_records"] == 0:
        return {
            "status": "Sem execução registrada",
            "interpretação": "Nenhum manifesto local foi encontrado.",
            "total": 0,
        }
    return {
        "status": "Operacional",
        "interpretação": "Há manifestos locais disponíveis para rastreabilidade.",
        "total": summary["total_records"],
        "status_counts": summary["status_counts"],
    }


def _privacy_indicator(label: str, metric: str, value: Any, record: HistoryRecord) -> dict[str, Any]:
    unavailable = value is None
    return {
        "indicador": label,
        "risco": "Não avaliado" if unavailable else "Moderado",
        "métrica": metric,
        "valor": "Não avaliado" if unavailable else value,
        "limite": "Ver quality gates do experimento",
        "fonte": record.identifier,
        "interpretação": "Indicador de risco, não prova automática de anonimização.",
        "data": record.created_at_utc or "Não disponível",
    }


def _empty_indicator(label: str, metric: str) -> dict[str, Any]:
    return {
        "indicador": label,
        "risco": "Não avaliado",
        "métrica": metric,
        "valor": "Não avaliado",
        "limite": "Não aplicável",
        "fonte": "Não disponível",
        "interpretação": "Não há evidência local suficiente para calcular este indicador.",
        "data": "Não disponível",
    }


def _find_metric(manifest: dict[str, Any], metric_name: str) -> Any:
    candidates = [
        manifest.get(metric_name),
        manifest.get("evaluation", {}).get(metric_name) if isinstance(manifest.get("evaluation"), dict) else None,
        manifest.get("summary", {}).get(metric_name) if isinstance(manifest.get("summary"), dict) else None,
    ]
    for value in candidates:
        if value is not None:
            return value
    return None
