"""Serviço de governança para indicadores e evidências da interface."""

from __future__ import annotations

import json
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
    diversity_memorization_indicators: list[dict[str, Any]]
    conditional_realism_indicators: list[dict[str, Any]]
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
        diversity_memorization_indicators=_diversity_memorization_indicators(history),
        conditional_realism_indicators=_conditional_realism_indicators(history),
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
                "modelo_de_renda": artifact.income_model_version,
                "vocabulário": artifact.categorical_vocabulary_version,
                "localidade": artifact.data_locale or "Não disponível",
                "normalização": artifact.unicode_normalization or "Não disponível",
                "propósito": artifact.purpose,
                "status": artifact.approval_status,
                "legado": "sim" if artifact.is_legacy_vocabulary else "não",
                "renda_legada": "sim" if artifact.is_legacy_income_model else "não",
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


def _diversity_memorization_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    latest = _latest_record_with_evaluation(history)
    if latest is None:
        return [
            _empty_indicator("CombinaÃ§Ãµes-base Ãºnicas", "privacy.unique_combinations"),
            _empty_indicator("Duplicidade de combinaÃ§Ãµes-base", "privacy.duplicate_base_rows.duplicate_row_rate"),
            _empty_indicator("CorrespondÃªncia exata com treino", "privacy.exact_matches.train.exact_match_rate"),
            _empty_indicator("CorrespondÃªncia exata com holdout", "privacy.exact_matches.holdout.exact_match_rate"),
        ]
    record, evaluation = latest
    privacy = evaluation.get("privacy", {}) if isinstance(evaluation.get("privacy"), dict) else {}
    duplicate_base = privacy.get("duplicate_base_rows") if isinstance(privacy.get("duplicate_base_rows"), dict) else {}
    exact_matches = privacy.get("exact_matches") if isinstance(privacy.get("exact_matches"), dict) else {}
    train = exact_matches.get("train") if isinstance(exact_matches.get("train"), dict) else {}
    holdout = exact_matches.get("holdout") if isinstance(exact_matches.get("holdout"), dict) else {}
    nearest = privacy.get("nearest_neighbor_train") if isinstance(privacy.get("nearest_neighbor_train"), dict) else {}
    dcr = nearest.get("distance_to_closest_record") if isinstance(nearest.get("distance_to_closest_record"), dict) else {}
    nndr = nearest.get("nearest_neighbor_distance_ratio") if isinstance(nearest.get("nearest_neighbor_distance_ratio"), dict) else {}
    return [
        _evidence_indicator(
            "CombinaÃ§Ãµes-base Ãºnicas",
            "privacy.unique_combinations",
            privacy.get("unique_combinations"),
            record,
            "Quantidade de combinaÃ§Ãµes distintas nas 11 colunas-base.",
            "evaluation.json â†’ privacy â†’ unique_combinations",
        ),
        _evidence_indicator(
            "Taxa de combinaÃ§Ãµes-base Ãºnicas",
            "privacy.unique_combination_rate",
            privacy.get("unique_combination_rate"),
            record,
            "ProporÃ§Ã£o de linhas sintÃ©ticas com combinaÃ§Ã£o-base distinta.",
            "evaluation.json â†’ privacy â†’ unique_combination_rate",
        ),
        _evidence_indicator(
            "OcorrÃªncias duplicadas",
            "privacy.duplicate_base_rows.duplicated_occurrences",
            duplicate_base.get("duplicated_occurrences"),
            record,
            "OcorrÃªncias posteriores Ã  primeira em grupos duplicados.",
            "evaluation.json â†’ privacy â†’ duplicate_base_rows",
        ),
        _evidence_indicator(
            "Grupos duplicados",
            "privacy.duplicate_base_rows.duplicated_groups",
            duplicate_base.get("duplicated_groups"),
            record,
            "CombinaÃ§Ãµes-base distintas que aparecem mais de uma vez.",
            "evaluation.json â†’ privacy â†’ duplicate_base_rows",
        ),
        _evidence_indicator(
            "Taxa de duplicidade",
            "privacy.duplicate_base_rows.duplicate_row_rate",
            duplicate_base.get("duplicate_row_rate"),
            record,
            "Duplicidade de combinaÃ§Ãµes-base; identificadores derivados nÃ£o participam.",
            "evaluation.json â†’ privacy â†’ duplicate_base_rows",
        ),
        _evidence_indicator(
            "CorrespondÃªncias exatas com treino",
            "privacy.exact_matches.train.exact_match_count",
            train.get("exact_match_count"),
            record,
            "Quantidade de perfis sintÃ©ticos cujas 11 colunas-base coincidem com o treino.",
            "evaluation.json â†’ privacy â†’ exact_matches â†’ train",
        ),
        _evidence_indicator(
            "Taxa de correspondÃªncia exata com treino",
            "privacy.exact_matches.train.exact_match_rate",
            train.get("exact_match_rate"),
            record,
            "Indicador de possÃ­vel memorizaÃ§Ã£o ou coincidÃªncia estatÃ­stica.",
            "evaluation.json â†’ privacy â†’ exact_matches â†’ train",
        ),
        _evidence_indicator(
            "CorrespondÃªncias exatas com holdout",
            "privacy.exact_matches.holdout.exact_match_count",
            holdout.get("exact_match_count"),
            record,
            "MÃ©trica de controle contra dados nÃ£o usados no treino.",
            "evaluation.json â†’ privacy â†’ exact_matches â†’ holdout",
        ),
        _evidence_indicator(
            "Taxa de correspondÃªncia exata com holdout",
            "privacy.exact_matches.holdout.exact_match_rate",
            holdout.get("exact_match_rate"),
            record,
            "Ajuda a distinguir memorizaÃ§Ã£o de coincidÃªncias da distribuiÃ§Ã£o.",
            "evaluation.json â†’ privacy â†’ exact_matches â†’ holdout",
        ),
        _evidence_indicator(
            "DistÃ¢ncia para o registro mais prÃ³ximo",
            "privacy.nearest_neighbor_train.distance_to_closest_record.mean",
            dcr.get("mean"),
            record,
            "DistÃ¢ncia mÃ©dia ao registro de treino mais prÃ³ximo nas colunas-base.",
            "evaluation.json â†’ privacy â†’ nearest_neighbor_train",
        ),
        _evidence_indicator(
            "NNDR",
            "privacy.nearest_neighbor_train.nearest_neighbor_distance_ratio.mean",
            nndr.get("mean"),
            record,
            "RazÃ£o de distÃ¢ncias entre o vizinho mais prÃ³ximo e o segundo mais prÃ³ximo.",
            "evaluation.json â†’ privacy â†’ nearest_neighbor_train",
        ),
    ]


def _conditional_realism_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    latest = _latest_record_with_evaluation(history)
    if latest is None:
        return [
            _empty_indicator("VersÃ£o do modelo de renda", "manifest.income_model_version"),
            _empty_indicator("Maior desvio condicional", "conditional_income.summary.max_conditional_income_wasserstein"),
        ]
    record, evaluation = latest
    conditional = evaluation.get("conditional_income", {}) if isinstance(evaluation.get("conditional_income"), dict) else {}
    summary = conditional.get("summary", {}) if isinstance(conditional.get("summary"), dict) else {}
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    return [
        _evidence_indicator(
            "VersÃ£o do modelo de renda",
            "manifest.income_model_version",
            manifest.get("income_model_version"),
            record,
            "VersÃ£o da calibraÃ§Ã£o sintÃ©tica usada para renda.",
            "manifest.json â†’ income_model_version",
        ),
        _evidence_indicator(
            "Grupos avaliados",
            "conditional_income.summary.conditional_groups_compared",
            summary.get("conditional_groups_compared"),
            record,
            "Quantidade de grupos condicionais com amostra suficiente.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
        _evidence_indicator(
            "Maior desvio condicional",
            "conditional_income.summary.max_conditional_income_wasserstein",
            summary.get("max_conditional_income_wasserstein"),
            record,
            "Maior distÃ¢ncia Wasserstein observada entre renda sintÃ©tica e referÃªncia dentro de grupos.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
        _evidence_indicator(
            "Maior diferenÃ§a de p95",
            "conditional_income.summary.max_abs_p95_difference",
            summary.get("max_abs_p95_difference"),
            record,
            "Maior diferenÃ§a absoluta no percentil 95 condicional.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
        _evidence_indicator(
            "Maior diferenÃ§a de p99",
            "conditional_income.summary.max_abs_p99_difference",
            summary.get("max_abs_p99_difference"),
            record,
            "Maior diferenÃ§a absoluta no percentil 99 condicional.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
        _evidence_indicator(
            "Grupos com cauda excessiva",
            "conditional_income.summary.groups_with_excessive_tail",
            summary.get("groups_with_excessive_tail"),
            record,
            "Grupos em que a cauda superior sintÃ©tica superou o limiar diagnÃ³stico.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
        _evidence_indicator(
            "Status da avaliaÃ§Ã£o",
            "conditional_income.summary.status",
            summary.get("status"),
            record,
            "SituaÃ§Ã£o diagnÃ³stica da avaliaÃ§Ã£o condicional.",
            "evaluation.json â†’ conditional_income â†’ summary",
        ),
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


def _evidence_indicator(
    label: str,
    metric: str,
    value: Any,
    record: HistoryRecord,
    interpretation: str,
    source: str,
) -> dict[str, Any]:
    unavailable = value is None
    return {
        "indicador": label,
        "risco": "NÃ£o avaliado" if unavailable else "DiagnÃ³stico",
        "mÃ©trica": metric,
        "valor": "NÃ£o avaliado" if unavailable else value,
        "unidade": "taxa" if str(metric).endswith("_rate") else "valor",
        "fonte": source,
        "execuÃ§Ã£o": record.identifier,
        "interpretaÃ§Ã£o": (
            "Esta execuÃ§Ã£o foi produzida antes da inclusÃ£o desta mÃ©trica ou nÃ£o contÃ©m os artefatos necessÃ¡rios."
            if unavailable
            else interpretation
        ),
        "data": record.created_at_utc or "NÃ£o disponÃ­vel",
    }


def _latest_record_with_evaluation(history: list[HistoryRecord]) -> tuple[HistoryRecord, dict[str, Any]] | None:
    for record in history:
        evaluation = _read_record_evaluation(record)
        if evaluation:
            return record, evaluation
    return None


def _read_record_evaluation(record: HistoryRecord) -> dict[str, Any]:
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    embedded = manifest.get("evaluation")
    if isinstance(embedded, dict) and embedded:
        return embedded
    sibling = record.path.parent / "evaluation.json"
    if not sibling.exists():
        return {}
    try:
        with sibling.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
