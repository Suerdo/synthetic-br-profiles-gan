"""Serviço compartilhado de governança para FastAPI, React e relatórios."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.domain.occupations import OCCUPATION_CATALOG
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, DATA_LOCALE, INCOME_MODEL_VERSION, UNICODE_NORMALIZATION
from synthetic_br_profiles_gan.models.registry import SavedModelArtifact, get_recommended_artifact, list_saved_model_artifacts
from synthetic_br_profiles_gan.services.audit_service import read_audit_events
from synthetic_br_profiles_gan.services.execution_history import (
    HistoryRecord,
    history_as_rows,
    history_summary,
    load_history,
    public_history_row,
)


NOT_EVALUATED = "Não avaliado"


@dataclass(frozen=True)
class GovernanceSourceConfig:
    """Configuração mínima para construir evidências de governança."""

    artifacts_root: Path
    models_root: Path
    audit_events_path: Path
    default_model: str = "programmatic"
    approved_model_artifacts: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class GovernanceSnapshot:
    """Conjunto de indicadores exibidos nas interfaces de governança."""

    overview: dict[str, Any]
    quality_indicators: list[dict[str, Any]]
    privacy_indicators: list[dict[str, Any]]
    diversity_memorization_indicators: list[dict[str, Any]]
    conditional_realism_indicators: list[dict[str, Any]]
    risk_indicators: list[dict[str, Any]]
    pipeline_status: dict[str, Any]
    model_versions: list[dict[str, Any]]
    recommended_neural_model: list[dict[str, Any]]
    history: list[HistoryRecord]
    audit_events: list[dict[str, Any]]


def build_governance_snapshot(config: Any) -> GovernanceSnapshot:
    """Monta indicadores de governança apenas com evidências locais disponíveis."""
    resolved = _resolve_config(config)
    history = load_history(resolved.artifacts_root)
    summary = history_summary(history)
    artifacts = list_saved_model_artifacts(resolved.models_root)
    recommended_ctgan = get_recommended_artifact(resolved.models_root, "ctgan")
    approved_generation_artifacts = [
        artifact for artifact in artifacts if artifact.model in {"ctgan", "simple_gan"} and is_approved_vocabulary_v2_artifact(artifact, resolved)
    ]
    latest_approved = _latest_artifact(approved_generation_artifacts)
    overview = {
        "default_model": default_generation_model(resolved, artifacts),
        "available_models": _available_model_labels(approved_generation_artifacts),
        "vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "income_model_version": INCOME_MODEL_VERSION,
        "data_locale": DATA_LOCALE,
        "unicode_normalization": UNICODE_NORMALIZATION,
        "total_runs": summary["total_records"],
        "latest_run": summary["latest_identifier"] or "Sem execução registrada",
        "latest_status": summary["latest_status"],
        "occupation_count": len(OCCUPATION_CATALOG),
        "pipeline_status": "Operacional" if summary["total_records"] > 0 else "Sem execução registrada",
        "latest_approved_model_version": None if latest_approved is None else latest_approved.artifact_id,
    }
    return GovernanceSnapshot(
        overview=overview,
        quality_indicators=_quality_indicators(history),
        privacy_indicators=_privacy_indicators(history),
        diversity_memorization_indicators=_diversity_memorization_indicators(history),
        conditional_realism_indicators=_conditional_realism_indicators(history),
        risk_indicators=_risk_indicators(history),
        pipeline_status=_pipeline_status(summary),
        model_versions=model_version_rows(artifacts, resolved),
        recommended_neural_model=_recommended_neural_model_rows(recommended_ctgan),
        history=history,
        audit_events=read_audit_events(resolved.audit_events_path, limit=200),
    )


def build_governance_api_snapshot(config: Any) -> dict[str, Any]:
    """Retorna um snapshot tipado e sanitizado para a API React."""
    resolved = _resolve_config(config)
    snapshot = build_governance_snapshot(resolved)
    artifacts = list_saved_model_artifacts(resolved.models_root)
    recommended_artifact = get_recommended_artifact(resolved.models_root, "ctgan")
    recommended = None if recommended_artifact is None else artifact_public_summary(recommended_artifact)
    executions = [public_history_row(record) for record in snapshot.history[:100]]
    evaluated_count = sum(1 for record in snapshot.history if _read_record_evaluation(record))
    latest_evaluation = next((record for record in snapshot.history if _read_record_evaluation(record)), None)
    operational_summary = dict(snapshot.overview)
    operational_summary.update(
        {
            "registered_executions": len(snapshot.history),
            "evaluated_executions": evaluated_count,
            "latest_evaluation": None if latest_evaluation is None else latest_evaluation.identifier,
            "strategy_count": 3,
        }
    )
    return {
        "governance_decision": governance_decision_summary(recommended_artifact),
        "available_strategies": available_strategy_rows(artifacts, resolved),
        "provenance": provenance_summary(recommended_artifact),
        "quality_gates": quality_gate_rows(recommended_artifact, snapshot.quality_indicators),
        "operational": {
            "metrics": [
                _metric("Estratégias disponíveis", "strategy_count", 3, "catálogo do projeto", "Programático, CTGAN e GAN simples existem como estratégias, independentemente de artefato neural."),
                _metric("Modelo padrão geral", "default_model", snapshot.overview["default_model"], "configs/api.yaml", "Modelo inicial da interface; não significa melhor desempenho universal."),
                _metric("Modelo neural recomendado", "recommended_neural_model", None if recommended is None else recommended["artifact_id"], "ModelRegistry", "Artefato neural aprovado internamente, quando disponível."),
                _metric("Execuções registradas", "registered_executions", len(snapshot.history), "manifestos em artifacts/", "Quantidade de registros operacionais identificados localmente."),
                _metric("Execuções com avaliação completa", "evaluated_executions", evaluated_count, "evaluation.json ou manifesto", "Quantidade de registros com avaliação agregada disponível."),
                _metric("Última avaliação", "latest_evaluation", None if latest_evaluation is None else latest_evaluation.identifier, "evaluation.json ou manifesto", "Registro avaliado mais recente identificado."),
                _metric("Status das evidências", "pipeline_status", snapshot.overview["pipeline_status"], "manifestos em artifacts/", "Indica se há evidências locais para análise."),
            ],
            "summary": operational_summary,
        },
        "recommended_model": recommended,
        "evidence_by_model": governance_evidence_by_model(snapshot.history, artifacts, recommended_artifact),
        "quality": {
            "indicators": snapshot.quality_indicators,
            "status": _section_status(snapshot.quality_indicators),
        },
        "privacy": {
            "indicators": snapshot.privacy_indicators,
            "diversity_memorization": snapshot.diversity_memorization_indicators,
            "status": _section_status(snapshot.diversity_memorization_indicators),
        },
        "income": {
            "indicators": snapshot.conditional_realism_indicators,
            "status": _section_status(snapshot.conditional_realism_indicators),
        },
        "executions": executions,
        "audit": snapshot.audit_events,
        "glossary": governance_glossary(),
    }


def governance_evidence_by_model(
    history: list[HistoryRecord],
    artifacts: list[SavedModelArtifact],
    recommended_artifact: SavedModelArtifact | None,
) -> dict[str, Any]:
    """Monta evidências sanitizadas por estratégia de modelo."""
    entries: list[dict[str, Any]] = []
    for model in ("programmatic", "ctgan", "simple_gan"):
        entries.append(_model_evidence_entry(model, history, artifacts, recommended_artifact))
    return {
        "default_model": "ctgan" if recommended_artifact is not None else "programmatic",
        "models": entries,
    }


def _model_evidence_entry(
    model: str,
    history: list[HistoryRecord],
    artifacts: list[SavedModelArtifact],
    recommended_artifact: SavedModelArtifact | None,
) -> dict[str, Any]:
    model_history = _history_for_model(history, model)
    latest_record = model_history[0] if model_history else None
    evaluation_record = _latest_record_with_evaluation(history, model=model)
    quality_record = _latest_record_with_quality_gates(history, model=model)
    artifact = recommended_artifact if model == "ctgan" else _latest_artifact([item for item in artifacts if item.model == model])
    if model == "ctgan" and recommended_artifact is not None:
        quality_gates = quality_gate_rows(recommended_artifact, [])
    else:
        quality_gates = _quality_gate_rows_from_record(None if quality_record is None else quality_record[0])
    if not quality_gates and model_history:
        quality_gates = quality_gate_rows(None, _quality_indicators(model_history))
    privacy = _diversity_memorization_indicators(history, model=model)
    income = _conditional_realism_indicators(history, model=model)
    evidence_status = _model_evidence_status(model, artifact, latest_record)
    return {
        "model": model,
        "label": _model_label(model),
        "status_label": evidence_status,
        "role": _model_evidence_role(model),
        "source_kind": None if latest_record is None else latest_record.kind,
        "source_identifier": _model_evidence_source_identifier(artifact, evaluation_record, latest_record),
        "latest_execution_status": None if latest_record is None else latest_record.status,
        "artifact_id": None if artifact is None else artifact.artifact_id,
        "has_evidence": bool(quality_gates or _has_available_indicator(privacy) or _has_available_indicator(income)),
        "quality_gates": quality_gates,
        "privacy": {
            "diversity_memorization": privacy,
            "status": _section_status(privacy),
        },
        "income": {
            "indicators": income,
            "status": _section_status(income),
        },
    }


def _model_label(model: str) -> str:
    return {"programmatic": "Programático", "ctgan": "CTGAN", "simple_gan": "GAN simples"}.get(model, model)


def _model_evidence_role(model: str) -> str:
    roles = {
        "programmatic": "Padrão geral",
        "ctgan": "Modelo neural recomendado",
        "simple_gan": "Experimental",
    }
    return roles.get(model, "Estratégia")


def _model_evidence_status(model: str, artifact: SavedModelArtifact | None, latest_record: HistoryRecord | None) -> str:
    if model == "programmatic":
        return "Padrão geral"
    if model == "simple_gan":
        status = latest_record.status if latest_record and latest_record.status else None
        return "Experimental" if status is None else f"Experimental · {status}"
    if artifact is not None and artifact.recommended_for_neural_generation:
        return "Recomendado"
    if artifact is not None:
        return artifact.approval_status or artifact.purpose or NOT_EVALUATED
    return NOT_EVALUATED


def _model_evidence_source_identifier(
    artifact: SavedModelArtifact | None,
    evaluation_record: tuple[HistoryRecord, dict[str, Any]] | None,
    latest_record: HistoryRecord | None,
) -> str | None:
    if artifact is not None and artifact.model == "ctgan" and artifact.recommended_for_neural_generation:
        return artifact.artifact_id
    if evaluation_record is not None:
        return evaluation_record[0].identifier
    if latest_record is not None:
        return latest_record.identifier
    return None


def _history_for_model(history: list[HistoryRecord], model: str) -> list[HistoryRecord]:
    return [record for record in history if record.model == model]


def _has_available_indicator(indicators: list[dict[str, Any]]) -> bool:
    return any(item.get("value") is not None and item.get("value") != NOT_EVALUATED for item in indicators)


def recommended_neural_model_public(models_root: str | Path) -> dict[str, Any] | None:
    artifact = get_recommended_artifact(models_root, "ctgan")
    if artifact is None:
        return None
    return artifact_public_summary(artifact)


def artifact_public_summary(artifact: SavedModelArtifact) -> dict[str, Any]:
    """Retorna resumo público e sanitizado de um artefato de modelo."""
    metrics = artifact_quality_summary(artifact)
    approval = _approval_manifest(artifact)
    return {
        "artifact_id": artifact.artifact_id,
        "model": artifact.model,
        "status": artifact.approval_status,
        "purpose": artifact.purpose,
        "evaluation_status": _evaluation_status(artifact),
        "recommendation_status": "recommended" if artifact.manifest.get("recommended_for_neural_generation") else "not_recommended",
        "general_default": bool(artifact.manifest.get("general_platform_default")),
        "production_status": "not_approved",
        "created_at_utc": artifact.created_at_utc,
        "approved_at_utc": approval.get("approved_at_utc") or artifact.manifest.get("approved_at_utc"),
        "vocabulary_version": artifact.categorical_vocabulary_version,
        "income_model_version": artifact.income_model_version,
        "geography_model_version": artifact.geography_model_version,
        "geography_catalog_checksum": artifact.geography_catalog_checksum,
        "benchmark": artifact.manifest.get("confirmation_benchmark"),
        "seeds": artifact.manifest.get("confirmation_seeds") if isinstance(artifact.manifest.get("confirmation_seeds"), list) else None,
        "result": _confirmation_result(artifact),
        "metrics": metrics,
        "limitations": _artifact_limitations(artifact),
        "environment": _sanitized_environment(artifact.manifest),
        "approval_note": (
            "A aprovação representa uma decisão técnica interna baseada nos critérios do projeto. "
            "Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
        ),
    }


def governance_decision_summary(artifact: SavedModelArtifact | None) -> dict[str, Any]:
    """Resume a decisão técnica interna sem expor caminhos locais."""
    if artifact is None:
        return {
            "title": "Decisão de Governança",
            "status_label": "Não avaliado",
            "evaluation_status": "not_evaluated",
            "recommendation_status": "not_recommended",
            "general_default": False,
            "production_status": "not_defined",
            "scope": "Nenhum artefato neural recomendado foi encontrado.",
            "decided_at_utc": None,
            "benchmark": None,
            "mandatory_passed": None,
            "mandatory_total": None,
            "caveats_count": None,
            "artifact_id": None,
            "model": None,
            "disclaimer": _approval_disclaimer(),
        }
    approval = _approval_manifest(artifact)
    mandatory = approval.get("mandatory_checks") if isinstance(approval.get("mandatory_checks"), dict) else {}
    mandatory_total = len(mandatory)
    mandatory_passed = sum(1 for item in mandatory.values() if isinstance(item, dict) and item.get("passed") is True)
    limitations = approval.get("known_limitations") if isinstance(approval.get("known_limitations"), list) else artifact.manifest.get("limitations")
    caveats_count = len(limitations) if isinstance(limitations, list) else 0
    benchmark = _first_benchmark_id(approval.get("evidence_benchmarks")) or artifact.manifest.get("confirmation_benchmark")
    approved = artifact.approval_status == "approved" or artifact.purpose == "approved"
    recommended = bool(artifact.manifest.get("recommended_for_neural_generation"))
    return {
        "title": "Decisão de Governança",
        "status_label": "Aprovado nos gates internos" if approved else "Não aprovado",
        "evaluation_status": "approved" if approved else (artifact.approval_status or artifact.purpose or "not_evaluated"),
        "recommendation_status": "recommended" if recommended else "not_recommended",
        "general_default": bool(artifact.manifest.get("general_platform_default")),
        "production_status": "not_approved",
        "scope": "Artefato neural CTGAN com vocabulário v2, renda v3 e geografia v2.",
        "decided_at_utc": approval.get("approved_at_utc") or artifact.manifest.get("approved_at_utc"),
        "benchmark": benchmark,
        "mandatory_passed": mandatory_passed if mandatory_total else None,
        "mandatory_total": mandatory_total or None,
        "caveats_count": caveats_count,
        "artifact_id": artifact.artifact_id,
        "model": artifact.model,
        "disclaimer": _approval_disclaimer(),
    }


def available_strategy_rows(artifacts: list[SavedModelArtifact], config: Any) -> list[dict[str, Any]]:
    """Lista as três estratégias do projeto, separando estratégia de artefato."""
    resolved = _resolve_config(config)
    by_model = {
        "ctgan": [artifact for artifact in artifacts if artifact.model == "ctgan"],
        "simple_gan": [artifact for artifact in artifacts if artifact.model == "simple_gan"],
    }
    recommended_ctgan = get_recommended_artifact(resolved.models_root, "ctgan")
    return [
        {
            "model": "programmatic",
            "label": "Programático",
            "role": "Padrão geral da plataforma",
            "status": "Disponível",
            "operational_availability": "Disponível",
            "artifact_required": False,
            "artifact_available": True,
            "artifact_count": 0,
            "recommended_artifact_id": None,
        },
        {
            "model": "ctgan",
            "label": "CTGAN",
            "role": "Modelo neural recomendado" if recommended_ctgan else "Estratégia neural avançada",
            "status": "Artefato aprovado disponível" if recommended_ctgan and recommended_ctgan.approval_status == "approved" else ("Artefato válido disponível" if by_model["ctgan"] else "Sem artefato válido"),
            "operational_availability": "Disponível" if by_model["ctgan"] else "Indisponível",
            "artifact_required": True,
            "artifact_available": bool(by_model["ctgan"]),
            "artifact_count": len(by_model["ctgan"]),
            "recommended_artifact_id": None if recommended_ctgan is None else recommended_ctgan.artifact_id,
        },
        {
            "model": "simple_gan",
            "label": "GAN simples",
            "role": "Baseline acadêmico experimental",
            "status": "Artefato válido disponível" if by_model["simple_gan"] else "Sem artefato válido",
            "operational_availability": "Disponível" if by_model["simple_gan"] else "Indisponível",
            "artifact_required": True,
            "artifact_available": bool(by_model["simple_gan"]),
            "artifact_count": len(by_model["simple_gan"]),
            "recommended_artifact_id": None,
        },
    ]


def provenance_summary(artifact: SavedModelArtifact | None) -> dict[str, Any]:
    """Agrega proveniência reprodutível sem hostname, usuário, paths ou hardware."""
    if artifact is None:
        return {"items": [], "groups": [], "timeline": [], "current_code_commit": _current_git_commit_short()}
    manifest = artifact.manifest
    approval = _approval_manifest(artifact)
    metrics = artifact_quality_summary(artifact)
    environment = _sanitized_environment(manifest)
    ctgan_config = manifest.get("ctgan_config") if isinstance(manifest.get("ctgan_config"), dict) else {}
    benchmark = _first_benchmark_id(approval.get("evidence_benchmarks")) or manifest.get("confirmation_benchmark")
    split_strategy = _split_strategy(manifest)
    items = [
        _provenance_item("Artefato", artifact.artifact_id, "training_manifest.json"),
        _provenance_item("Benchmark", benchmark, "approval_manifest.json"),
        _provenance_item("Data do artefato", manifest.get("created_at_utc"), "training_manifest.json"),
        _provenance_item("Data da decisão", approval.get("approved_at_utc") or manifest.get("approved_at_utc"), "approval_manifest.json"),
        _provenance_item("Seeds", manifest.get("confirmation_seeds"), "approval_manifest.json"),
        _provenance_item("Commit atual da aplicação", _current_git_commit_short(), ".git/HEAD"),
        _provenance_item("Treino", manifest.get("train_rows"), "training_manifest.json"),
        _provenance_item("Holdout", manifest.get("holdout_rows"), "training_manifest.json"),
        _provenance_item("Calibração", manifest.get("calibration_rows"), "training_manifest.json"),
        _provenance_item("Estratégia de split", split_strategy, "training_manifest.json"),
        _provenance_item("Épocas", ctgan_config.get("epochs") or metrics.get("epochs"), "ctgan_config"),
        _provenance_item("Batch size", ctgan_config.get("batch_size"), "ctgan_config"),
        _provenance_item("PAC", ctgan_config.get("pac"), "ctgan_config"),
        _provenance_item("Generator LR", ctgan_config.get("generator_lr"), "ctgan_config"),
        _provenance_item("Discriminator LR", ctgan_config.get("discriminator_lr"), "ctgan_config"),
        _provenance_item("Python", environment.get("python"), "training_manifest.json → environment"),
        _provenance_item("CTGAN", environment.get("ctgan"), "training_manifest.json → environment.library_versions"),
        _provenance_item("TensorFlow", environment.get("tensorflow"), "training_manifest.json → environment.library_versions"),
        _provenance_item("Vocabulário", manifest.get("categorical_vocabulary_version"), "training_manifest.json"),
        _provenance_item("Renda", manifest.get("income_model_version"), "training_manifest.json"),
        _provenance_item("Geografia", manifest.get("geography_model_version"), "training_manifest.json"),
        _provenance_item("Checksum geográfico", manifest.get("geography_catalog_checksum"), "training_manifest.json"),
        _provenance_item("Tamanho do modelo", manifest.get("model_size_bytes"), "training_manifest.json"),
    ]
    return {
        "items": items,
        "groups": [
            {"title": "Identificação", "keys": ["Artefato", "Benchmark", "Data do artefato", "Data da decisão", "Seeds", "Commit atual da aplicação"]},
            {"title": "Dados e split", "keys": ["Treino", "Holdout", "Calibração", "Estratégia de split"]},
            {"title": "Hiperparâmetros", "keys": ["Épocas", "Batch size", "PAC", "Generator LR", "Discriminator LR"]},
            {"title": "Ambiente e versões", "keys": ["Python", "CTGAN", "TensorFlow", "Vocabulário", "Renda", "Geografia", "Checksum geográfico", "Tamanho do modelo"]},
        ],
        "timeline": [
            {"step": "Configuração", "source": "ctgan_config", "value": manifest.get("source_profile") or "ctgan_income_v3_geo_v2_candidate"},
            {"step": "Geração", "source": "benchmark", "value": benchmark},
            {"step": "Validação", "source": "approval_manifest.json", "value": f"{metrics.get('invalid_rows')} linhas finais inválidas"},
            {"step": "Avaliação", "source": "evaluation.json", "value": f"{metrics.get('approved_confirmation_seeds')}/{len(metrics.get('confirmation_seeds') or [])} seeds aprovadas" if metrics.get("confirmation_seeds") else None},
            {"step": "Decisão", "source": "approval_manifest.json", "value": artifact.approval_status},
            {"step": "Artefato", "source": "training_manifest.json", "value": artifact.artifact_id},
            {"step": "Rastreabilidade", "source": "manifestos", "value": "manifestos e checksums sanitizados"},
        ],
    }


def quality_gate_rows(artifact: SavedModelArtifact | None, fallback_indicators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normaliza checks de qualidade para tabela semântica da interface."""
    rows: list[dict[str, Any]] = []
    if artifact is not None:
        approval = _approval_manifest(artifact)
        mandatory = approval.get("mandatory_checks") if isinstance(approval.get("mandatory_checks"), dict) else {}
        for key, payload in mandatory.items():
            if not isinstance(payload, dict):
                continue
            observed, operator, threshold = _gate_observed_operator_threshold(key, payload.get("value"))
            rows.append(
                {
                    "id": key,
                    "metric": _humanize_gate_key(key),
                    "observed": observed,
                    "operator": operator,
                    "threshold": threshold,
                    "mandatory": True,
                    "passed": bool(payload.get("passed")),
                    "status": "Aprovado" if payload.get("passed") else "Reprovado",
                    "source": "approval_manifest.json",
                    "evidence": _sanitize_public_value(payload.get("value")),
                }
            )
    if rows:
        return rows
    for indicator in fallback_indicators:
        rows.append(
            {
                "id": indicator.get("metric") or indicator.get("métrica") or indicator.get("indicator"),
                "metric": indicator.get("indicator") or indicator.get("indicador"),
                "observed": indicator.get("value"),
                "operator": "=" if indicator.get("limit") is not None else None,
                "threshold": indicator.get("limit"),
                "mandatory": indicator.get("gate_type") == "mandatory",
                "passed": None if indicator.get("value") is None else True,
                "status": "Não avaliado" if indicator.get("value") is None else "Aprovado",
                "source": indicator.get("source") or indicator.get("fonte"),
                "evidence": None,
            }
        )
    return rows


def artifact_quality_summary(artifact: SavedModelArtifact) -> dict[str, Any]:
    """Extrai métricas públicas agregadas de um artefato, preservando `None` para ausências."""
    manifest = artifact.manifest if isinstance(artifact.manifest, dict) else {}
    approval = manifest.get("approval_evidence_summary") if isinstance(manifest.get("approval_evidence_summary"), dict) else {}
    aggregate = manifest.get("aggregate_metrics") if isinstance(manifest.get("aggregate_metrics"), dict) else {}
    metrics_by_seed = manifest.get("metrics_by_seed") if isinstance(manifest.get("metrics_by_seed"), list) else []
    raw_global_values = _seed_values(approval, "raw_global_validity_rate") or _row_values(metrics_by_seed, "raw_global_validity_rate")
    raw_geo_values = _seed_values(approval, "raw_geographic_validity_rate") or _row_values(metrics_by_seed, "raw_geographic_validity_rate")
    duplicate_base_rates = _seed_values(approval, "duplicate_base_row_rate") or _row_values(metrics_by_seed, "duplicate_base_row_rate")
    exact_train_rates = _seed_values(approval, "exact_train_match_rate") or _row_values(metrics_by_seed, "exact_train_match_rate")
    duplicated_identifiers = _seed_values(approval, "duplicated_identifiers")
    invalid_rows = _seed_values(approval, "invalid_rows") or _row_values(metrics_by_seed, "invalid_rows")
    return {
        "confirmation_seeds": manifest.get("confirmation_seeds") if isinstance(manifest.get("confirmation_seeds"), list) else None,
        "approved_confirmation_seeds": aggregate.get("approved_confirmation_seeds") or approval.get("approved_runs"),
        "raw_geographic_validity_rate": _range_or_single(raw_geo_values),
        "raw_global_validity_rate": _range_or_single(raw_global_values),
        "known_geography_key_rate": aggregate.get("known_geography_key_rate_min") or _min_or_none(_seed_values(approval, "known_geography_key_rate")),
        "state_coverage": _min_or_none(_seed_values(approval, "state_coverage")),
        "municipality_coverage": _min_or_none(_seed_values(approval, "municipality_coverage")),
        "ddd_coverage": _min_or_none(_seed_values(approval, "ddd_coverage")),
        "geography_key_coverage": aggregate.get("geography_key_coverage_min") or _min_or_none(_seed_values(approval, "geography_key_coverage")),
        "duplicate_base_row_rate": aggregate.get("duplicate_base_row_rate_max") if aggregate.get("duplicate_base_row_rate_max") is not None else _max_or_none(duplicate_base_rates),
        "duplicate_base_duplicated_occurrences": _max_or_none(_seed_values(approval, "duplicate_base_duplicated_occurrences")),
        "duplicated_identifiers": _max_or_none(duplicated_identifiers),
        "invalid_rows": _max_or_none(invalid_rows),
        "exact_train_match_rate": aggregate.get("exact_train_match_rate_max") if aggregate.get("exact_train_match_rate_max") is not None else _max_or_none(exact_train_rates),
        "exact_train_match_count": _max_or_none(_seed_values(approval, "exact_train_match_count")),
        "conditional_income_status": _conditional_income_status(manifest),
        "quality_status": aggregate.get("status") or artifact.approval_status or artifact.purpose,
        "training_seconds": _range_or_single(_row_values(metrics_by_seed, "training_seconds")),
        "generation_seconds": _range_or_single(_row_values(metrics_by_seed, "generation_seconds")),
        "peak_memory_mb": _range_or_single(_row_values(metrics_by_seed, "peak_memory_mb")),
        "model_size_mb": _range_or_single(_row_values(metrics_by_seed, "model_size_mb")),
        "library": _library_label(manifest),
        "epochs": _epochs(manifest),
    }


def is_approved_vocabulary_v2_artifact(artifact: SavedModelArtifact, config: Any) -> bool:
    """Indica se um artefato neural pode ser usado diretamente na geração pela interface."""
    resolved = _resolve_config(config)
    if artifact.model not in {"ctgan", "simple_gan"}:
        return False
    configured = artifact.artifact_id in set((resolved.approved_model_artifacts or {}).get(artifact.model, ()))
    approved = artifact.approval_status == "approved" or artifact.purpose == "approved" or configured
    return (
        approved
        and artifact.categorical_vocabulary_version >= CATEGORICAL_VOCABULARY_VERSION
        and artifact.data_locale == DATA_LOCALE
        and artifact.unicode_normalization == UNICODE_NORMALIZATION
    )


def default_generation_model(config: Any, artifacts: list[SavedModelArtifact] | None = None) -> str:
    """Escolhe o modelo padrão operacional da interface."""
    resolved = _resolve_config(config)
    if resolved.default_model == "programmatic":
        return "programmatic"
    materialized = artifacts if artifacts is not None else list_saved_model_artifacts(resolved.models_root)
    if resolved.default_model == "ctgan" and any(
        artifact.model == "ctgan" and is_approved_vocabulary_v2_artifact(artifact, resolved)
        for artifact in materialized
    ):
        return "ctgan"
    return "programmatic"


def model_version_rows(artifacts: list[SavedModelArtifact], config: Any) -> list[dict[str, Any]]:
    """Converte artefatos em linhas de histórico de modelos."""
    resolved = _resolve_config(config)
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        generation_ready = is_approved_vocabulary_v2_artifact(artifact, resolved)
        rows.append(
            {
                "modelo": artifact.model,
                "artefato": artifact.artifact_id,
                "criado_em_utc": artifact.created_at_utc or NOT_EVALUATED,
                "treino": artifact.train_rows if artifact.train_rows is not None else NOT_EVALUATED,
                "seed": artifact.seed if artifact.seed is not None else NOT_EVALUATED,
                "schema": artifact.schema_version,
                "modelo_de_renda": artifact.income_model_version,
                "vocabulário": artifact.categorical_vocabulary_version,
                "localidade": artifact.data_locale or NOT_EVALUATED,
                "normalização": artifact.unicode_normalization or NOT_EVALUATED,
                "propósito": artifact.purpose,
                "status": artifact.approval_status,
                "legado": "sim" if artifact.is_legacy_vocabulary else "não",
                "renda_legada": "sim" if artifact.is_legacy_income_model else "não",
                "disponível_para_geração": "sim" if generation_ready else "não",
            }
        )
    return rows


def governance_glossary() -> list[dict[str, str]]:
    """Retorna glossário operacional reutilizável pela API e frontend."""
    return [
        _glossary("Duplicidade de combinações-base", "Repetição exata das 11 colunas-base produzidas pelo modelo. Identificadores derivados não participam."),
        _glossary("Correspondência exata com treino", "Percentual de registros sintéticos cujas 11 colunas-base coincidem com pelo menos um registro de treinamento."),
        _glossary("Correspondência exata com holdout", "Métrica de controle com dados não usados no treino, útil para distinguir memorização de coincidências da distribuição."),
        _glossary("Realismo condicional", "Capacidade de preservar distribuições em contextos específicos, como renda por ocupação, escolaridade e idade."),
        _glossary("Cauda superior", "Região dos valores mais altos da distribuição, avaliada por percentis como p95 e p99."),
        _glossary("TVD", "Distância de variação total entre distribuições categóricas. Quanto menor, mais próximas estão as distribuições."),
        _glossary("DCR", "Distance to Closest Record: distância do registro sintético ao registro de referência mais próximo."),
        _glossary("NNDR", "Nearest Neighbor Distance Ratio: razão entre distâncias ao primeiro e ao segundo vizinho mais próximo."),
        _glossary("Validade estrutural", "Conjunto de regras de domínio, nulidade, consistência e relacionamento aplicadas ao schema final."),
        _glossary("Geo_Key", "Chave categórica interna que representa combinações permitidas de região, estado, município e DDD para CTGAN geography v2."),
    ]


def _recommended_neural_model_rows(artifact: SavedModelArtifact | None) -> list[dict[str, Any]]:
    if artifact is None:
        return [_recommended_row("Modelo neural recomendado", NOT_EVALUATED, "training_manifest.json", "Nenhum artefato neural aprovado e recomendado foi encontrado.")]
    public = artifact_public_summary(artifact)
    metrics = public["metrics"]
    return [
        _recommended_row("Identificador", public["artifact_id"], "training_manifest.json", "Diretório administrado do artefato aprovado."),
        _recommended_row("Status", public["status"], "training_manifest.json", "Status técnico interno do artefato."),
        _recommended_row("Vocabulário", f"v{public['vocabulary_version']}", "training_manifest.json", "Versão do vocabulário categórico."),
        _recommended_row("Renda", f"v{public['income_model_version']}", "training_manifest.json", "Versão da calibração sintética de renda."),
        _recommended_row("Geografia", f"v{public['geography_model_version']}", "training_manifest.json", "Versão da representação geográfica neural."),
        _recommended_row("Checksum geográfico", public.get("geography_catalog_checksum") or NOT_EVALUATED, "training_manifest.json", "Checksum do catálogo determinístico de Geo_Key."),
        _recommended_row("Benchmark de confirmação", public.get("benchmark") or NOT_EVALUATED, "approval_manifest.json", "Benchmark usado como evidência de aprovação."),
        _recommended_row("Seeds", ", ".join(str(seed) for seed in public.get("seeds") or []) or NOT_EVALUATED, "approval_manifest.json", "Seeds independentes da confirmação."),
        _recommended_row("Resultado", public.get("result") or NOT_EVALUATED, "run_summary.csv", "Resultado agregado da confirmação."),
        _recommended_row("Validade geográfica raw", _display_value(metrics.get("raw_geographic_validity_rate")), "run_summary.csv", "Validade geográfica bruta após decodificação de Geo_Key."),
        _recommended_row("Validade global raw", _display_value(metrics.get("raw_global_validity_rate")), "run_summary.csv", "Intervalo da validade estrutural bruta."),
        _recommended_row("Duplicidade-base", _display_value(metrics.get("duplicate_base_row_rate")), "evaluation.json → privacy", "Taxa de duplicidade das colunas-base."),
        _recommended_row("Match treino", _display_value(metrics.get("exact_train_match_rate")), "evaluation.json → privacy", "Correspondência exata com treino nas colunas-base."),
        _recommended_row("Cobertura", _coverage_label(metrics), "results.csv", "Cobertura geográfica e ocupacional disponível."),
        _recommended_row("Ambiente", public["environment"].get("summary") or NOT_EVALUATED, "training_manifest.json", "Ambiente sanitizado registrado no artefato aprovado."),
        _recommended_row("Limitações", "; ".join(public.get("limitations") or []) or NOT_EVALUATED, "approval_manifest.json", "Limitações conhecidas preservadas na aprovação."),
        _recommended_row("Nota", public["approval_note"], "approval_manifest.json", "Interpretação institucional da aprovação."),
    ]


def _recommended_row(campo: str, valor: Any, fonte: str, interpretacao: str) -> dict[str, Any]:
    return {"campo": campo, "valor": valor, "fonte": fonte, "interpretação": interpretacao}


def _approval_manifest(artifact: SavedModelArtifact) -> dict[str, Any]:
    path = artifact.artifact_path / "approval_manifest.json"
    try:
        with path.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        loaded = artifact.manifest.get("approval_manifest")
    return loaded if isinstance(loaded, dict) else {}


def _approval_disclaimer() -> str:
    return (
        "A aprovação representa uma decisão técnica interna baseada nos critérios do projeto. "
        "Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
    )


def _evaluation_status(artifact: SavedModelArtifact) -> str:
    if artifact.approval_status == "approved" or artifact.purpose == "approved":
        return "approved"
    if artifact.approval_status:
        return str(artifact.approval_status)
    return str(artifact.purpose or "not_evaluated")


def _first_benchmark_id(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return _basename_or_identifier(value[0])
    if isinstance(value, str):
        return _basename_or_identifier(value)
    return None


def _basename_or_identifier(value: Any) -> str:
    text = str(value)
    normalized = text.replace("\\", "/").rstrip("/")
    if normalized.startswith("artifacts/") or re.search(r"^[A-Za-z]:/", normalized):
        return normalized.split("/")[-1]
    return normalized


def _provenance_item(label: str, value: Any, source: str) -> dict[str, Any]:
    return {"label": label, "value": _sanitize_public_value(value), "source": source}


def _current_git_commit_short() -> str | None:
    git_root = Path.cwd() / ".git"
    head = git_root / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if content.startswith("ref:"):
        ref = content.split(":", 1)[1].strip()
        try:
            content = (git_root / ref).read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return content[:12] if re.fullmatch(r"[0-9a-fA-F]{40}", content) else None


def _split_strategy(manifest: dict[str, Any]) -> str | None:
    train = manifest.get("train_rows")
    holdout = manifest.get("holdout_rows")
    calibration = manifest.get("calibration_rows")
    if isinstance(train, int) and isinstance(holdout, int) and isinstance(calibration, int) and calibration:
        fraction = holdout / calibration
        return f"holdout_fraction={fraction:.2f}"
    return None


def _gate_observed_operator_threshold(key: str, value: Any) -> tuple[Any, str | None, Any]:
    lower = key.lower()
    observed = _extract_gate_observed(lower, value)
    if "zero" in lower:
        return observed, "=", 0
    if any(part in lower for part in ("coverage", "known_geography_key_rate", "raw_geographic_validity")):
        return observed, ">=", 1.0
    if "version" in lower:
        return observed, "=", observed
    if "approved" in lower:
        return observed, "=", "approved"
    return observed, None, None


def _extract_gate_observed(key: str, value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if not isinstance(value, dict):
        return _sanitize_public_value(value)
    if "zero_duplicate_base" in key:
        return value.get("duplicate_base_duplicated_occurrences") if value.get("duplicate_base_duplicated_occurrences") is not None else value.get("duplicate_base_row_rate")
    if "zero_exact_train_match" in key:
        return value.get("exact_train_match_count") if value.get("exact_train_match_count") is not None else value.get("exact_train_match_rate")
    if "zero_duplicated_identifiers" in key:
        return value.get("duplicated_identifiers")
    if "zero_invalid_rows" in key:
        return value.get("invalid_rows")
    for candidate in ("value", "status", "passed", "raw_geographic_validity_rate", "known_geography_key_rate", "coverage_rate"):
        if value.get(candidate) is not None:
            return value.get(candidate)
    return _sanitize_public_value(value)


def _humanize_gate_key(key: str) -> str:
    explicit = {
        "artifact_exists": "Artefato existente",
        "benchmark_exists": "Benchmark de confirmação existente",
        "model_is_ctgan": "Modelo CTGAN",
        "model_loadable": "Modelo carregável",
        "categorical_vocabulary_version": "Vocabulário v2",
        "income_model_version": "Renda v3",
        "geography_model_version": "Geografia v2",
        "geography_catalog_checksum": "Checksum geográfico",
        "external_schema_preserved": "Schema externo preservado",
        "geo_key_absent_from_public_schema": "Geo_Key ausente da saída pública",
        "occupational_coverage_documented": "Cobertura ocupacional documentada",
    }
    if key in explicit:
        return explicit[key]
    cleaned = re.sub(r"^seed_(\d+)_", r"Seed \1: ", key)
    cleaned = cleaned.replace("_", " ")
    return cleaned[:1].upper() + cleaned[1:]


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"hostname", "username", "user", "ip", "user_agent", "traceback", "cpu_count", "memory_gb", "processor", "platform"}:
                continue
            sanitized[str(key)] = _sanitize_public_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        return _basename_or_identifier(value)
    return value


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
        _indicator(
            "Validação estrutural",
            "validation.is_valid",
            bool(is_valid) if is_valid is not None else None,
            latest_pipeline.identifier,
            "O schema completo foi validado antes da exportação.",
            "validation.json ou manifesto da geração",
            risk="Baixo" if is_valid else "Elevado",
            limit=True,
            gate_type="mandatory",
            date=latest_pipeline.created_at_utc,
        ),
        _indicator(
            "Erros estruturais",
            "validation.reason_counts",
            sum(int(value) for value in reason_counts.values()) if isinstance(reason_counts, dict) else None,
            latest_pipeline.identifier,
            "Contagem agregada de violações estruturais do relatório mais recente.",
            "validation.json ou manifesto da geração",
            risk="Baixo" if not reason_counts else "Moderado",
            limit=0,
            gate_type="mandatory",
            date=latest_pipeline.created_at_utc,
        ),
    ]


def _privacy_indicators(history: list[HistoryRecord], *, model: str | None = None) -> list[dict[str, Any]]:
    latest = _latest_record_with_evaluation(history, model=model)
    if latest is None:
        return [_empty_indicator("Duplicidade de linhas", "duplicate_row_rate"), _empty_indicator("Correspondência exata com treino", "exact_train_match_rate")]
    record, evaluation = latest
    privacy = evaluation.get("privacy", {}) if isinstance(evaluation.get("privacy"), dict) else {}
    return [
        _indicator("Duplicidade de linhas", "privacy.duplicate_row_rate", privacy.get("duplicate_row_rate"), record.identifier, "Indicador de diversidade nas colunas-base.", "evaluation.json → privacy", risk="Diagnóstico", limit="quality gates", date=record.created_at_utc),
        _indicator("Correspondência exata com treino", "privacy.exact_train_match_rate", privacy.get("exact_train_match_rate"), record.identifier, "Indicador de possível memorização ou coincidência estatística.", "evaluation.json → privacy", risk="Diagnóstico", limit="quality gates", gate_type="mandatory", date=record.created_at_utc),
    ]


def _diversity_memorization_indicators(history: list[HistoryRecord], *, model: str | None = None) -> list[dict[str, Any]]:
    latest = _latest_record_with_evaluation(history, model=model)
    if latest is None:
        return [
            _empty_indicator("Combinações-base únicas", "privacy.unique_combinations"),
            _empty_indicator("Duplicidade de combinações-base", "privacy.duplicate_base_rows.duplicate_row_rate"),
            _empty_indicator("Correspondência exata com treino", "privacy.exact_matches.train.exact_match_rate"),
            _empty_indicator("Correspondência exata com holdout", "privacy.exact_matches.holdout.exact_match_rate"),
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
        _indicator("Combinações-base únicas", "privacy.unique_combinations", privacy.get("unique_combinations"), record.identifier, "Quantidade de combinações distintas nas 11 colunas-base.", "evaluation.json → privacy → unique_combinations", date=record.created_at_utc),
        _indicator("Taxa de combinações-base únicas", "privacy.unique_combination_rate", privacy.get("unique_combination_rate"), record.identifier, "Proporção de linhas sintéticas com combinação-base distinta.", "evaluation.json → privacy → unique_combination_rate", unit="taxa", date=record.created_at_utc),
        _indicator("Ocorrências duplicadas", "privacy.duplicate_base_rows.duplicated_occurrences", duplicate_base.get("duplicated_occurrences"), record.identifier, "Ocorrências posteriores à primeira em grupos duplicados.", "evaluation.json → privacy → duplicate_base_rows", date=record.created_at_utc),
        _indicator("Grupos duplicados", "privacy.duplicate_base_rows.duplicated_groups", duplicate_base.get("duplicated_groups"), record.identifier, "Combinações-base distintas que aparecem mais de uma vez.", "evaluation.json → privacy → duplicate_base_rows", date=record.created_at_utc),
        _indicator("Taxa de duplicidade", "privacy.duplicate_base_rows.duplicate_row_rate", duplicate_base.get("duplicate_row_rate"), record.identifier, "Duplicidade de combinações-base; identificadores derivados não participam.", "evaluation.json → privacy → duplicate_base_rows", unit="taxa", date=record.created_at_utc),
        _indicator("Correspondências exatas com treino", "privacy.exact_matches.train.exact_match_count", train.get("exact_match_count"), record.identifier, "Quantidade de perfis sintéticos cujas 11 colunas-base coincidem com o treino.", "evaluation.json → privacy → exact_matches → train", date=record.created_at_utc),
        _indicator("Taxa de correspondência exata com treino", "privacy.exact_matches.train.exact_match_rate", train.get("exact_match_rate"), record.identifier, "Indicador de possível memorização ou coincidência estatística.", "evaluation.json → privacy → exact_matches → train", unit="taxa", gate_type="mandatory", date=record.created_at_utc),
        _indicator("Correspondências exatas com holdout", "privacy.exact_matches.holdout.exact_match_count", holdout.get("exact_match_count"), record.identifier, "Métrica de controle contra dados não usados no treino.", "evaluation.json → privacy → exact_matches → holdout", date=record.created_at_utc),
        _indicator("Taxa de correspondência exata com holdout", "privacy.exact_matches.holdout.exact_match_rate", holdout.get("exact_match_rate"), record.identifier, "Ajuda a distinguir memorização de coincidências da distribuição.", "evaluation.json → privacy → exact_matches → holdout", unit="taxa", date=record.created_at_utc),
        _indicator("DCR", "privacy.nearest_neighbor_train.distance_to_closest_record.mean", dcr.get("mean"), record.identifier, "Distância média ao registro de treino mais próximo nas colunas-base.", "evaluation.json → privacy → nearest_neighbor_train", date=record.created_at_utc),
        _indicator("NNDR", "privacy.nearest_neighbor_train.nearest_neighbor_distance_ratio.mean", nndr.get("mean"), record.identifier, "Razão de distâncias entre o vizinho mais próximo e o segundo mais próximo.", "evaluation.json → privacy → nearest_neighbor_train", date=record.created_at_utc),
    ]


def _conditional_realism_indicators(history: list[HistoryRecord], *, model: str | None = None) -> list[dict[str, Any]]:
    latest = _latest_record_with_evaluation(history, model=model)
    if latest is None:
        return [
            _empty_indicator("Versão do modelo de renda", "manifest.income_model_version"),
            _empty_indicator("Maior desvio condicional", "conditional_income.summary.max_conditional_income_wasserstein"),
        ]
    record, evaluation = latest
    conditional = evaluation.get("conditional_income", {}) if isinstance(evaluation.get("conditional_income"), dict) else {}
    summary = conditional.get("summary", {}) if isinstance(conditional.get("summary"), dict) else {}
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    return [
        _indicator("Versão do modelo de renda", "manifest.income_model_version", manifest.get("income_model_version"), record.identifier, "Versão da calibração sintética usada para renda.", "manifest.json → income_model_version", date=record.created_at_utc),
        _indicator("Grupos avaliados", "conditional_income.summary.conditional_groups_compared", summary.get("conditional_groups_compared"), record.identifier, "Quantidade de grupos condicionais com amostra suficiente.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
        _indicator("Maior desvio condicional", "conditional_income.summary.max_conditional_income_wasserstein", summary.get("max_conditional_income_wasserstein"), record.identifier, "Maior distância Wasserstein observada entre renda sintética e referência dentro de grupos.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
        _indicator("Maior diferença de p95", "conditional_income.summary.max_abs_p95_difference", summary.get("max_abs_p95_difference"), record.identifier, "Maior diferença absoluta no percentil 95 condicional.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
        _indicator("Maior diferença de p99", "conditional_income.summary.max_abs_p99_difference", summary.get("max_abs_p99_difference"), record.identifier, "Maior diferença absoluta no percentil 99 condicional.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
        _indicator("Grupos com cauda elevada", "conditional_income.summary.groups_with_excessive_tail", summary.get("groups_with_excessive_tail"), record.identifier, "Grupos em que a cauda superior sintética superou o limiar diagnóstico.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
        _indicator("Status da avaliação", "conditional_income.summary.status", summary.get("status"), record.identifier, "Situação diagnóstica da avaliação condicional.", "evaluation.json → conditional_income → summary", date=record.created_at_utc),
    ]


def _risk_indicators(history: list[HistoryRecord]) -> list[dict[str, Any]]:
    latest = history[0] if history else None
    if latest is None:
        return [
            _indicator(
                "Risco geral operacional",
                "manifest.status",
                "Sem execução registrada",
                None,
                "Nenhum manifesto local foi encontrado para classificar risco operacional.",
                "manifestos em artifacts/",
                risk=NOT_EVALUATED,
                date=None,
            )
        ]
    status = latest.status
    risk = "Baixo" if status in {"approved", "completed"} else "Moderado" if "quarantine" in str(status).lower() or status is None else "Elevado"
    return [_indicator("Risco geral operacional", "manifest.status", status, latest.identifier, "Leitura conservadora do status mais recente; não é certificação de conformidade.", "manifesto de execução", risk=risk, limit="approved/completed para baixo risco técnico", date=latest.created_at_utc)]


def _pipeline_status(summary: dict[str, Any]) -> dict[str, Any]:
    if summary["total_records"] == 0:
        return {"status": "Sem execução registrada", "interpretação": "Nenhum manifesto local foi encontrado.", "total": 0}
    return {"status": "Operacional", "interpretação": "Há manifestos locais disponíveis para rastreabilidade.", "total": summary["total_records"], "status_counts": summary["status_counts"]}


def _indicator(
    label: str,
    metric: str,
    value: Any,
    source_record: str | None,
    interpretation: str,
    source: str,
    *,
    risk: str = "Diagnóstico",
    limit: Any = None,
    unit: str | None = None,
    gate_type: str = "informational",
    date: str | None = None,
) -> dict[str, Any]:
    return {
        "indicator": label,
        "indicador": label,
        "risk": NOT_EVALUATED if value is None else risk,
        "risco": NOT_EVALUATED if value is None else risk,
        "metric": metric,
        "métrica": metric,
        "value": value,
        "valor": value,
        "unit": unit or ("taxa" if metric.endswith("_rate") else "valor"),
        "unidade": unit or ("taxa" if metric.endswith("_rate") else "valor"),
        "limit": limit,
        "limite": limit,
        "gate_type": gate_type,
        "fonte": source,
        "source": source,
        "execution": source_record,
        "execução": source_record,
        "interpretation": (
            "Esta execução foi produzida antes da inclusão desta métrica ou não contém os artefatos necessários."
            if value is None
            else interpretation
        ),
        "interpretação": (
            "Esta execução foi produzida antes da inclusão desta métrica ou não contém os artefatos necessários."
            if value is None
            else interpretation
        ),
        "date": date,
        "data": date,
    }


def _empty_indicator(label: str, metric: str) -> dict[str, Any]:
    indicator = _indicator(label, metric, None, None, "Não há evidência local suficiente para calcular este indicador.", "Não disponível", risk=NOT_EVALUATED, limit=None, date=None)
    indicator["valor"] = NOT_EVALUATED
    return indicator


def _latest_record_with_evaluation(history: list[HistoryRecord], *, model: str | None = None) -> tuple[HistoryRecord, dict[str, Any]] | None:
    for record in history:
        if model is not None and record.model != model:
            continue
        evaluation = _read_record_evaluation(record)
        if evaluation:
            return record, evaluation
    return None


def _latest_record_with_quality_gates(history: list[HistoryRecord], *, model: str | None = None) -> tuple[HistoryRecord, dict[str, Any]] | None:
    for record in history:
        if model is not None and record.model != model:
            continue
        quality_gates = _read_record_quality_gates(record)
        if quality_gates:
            return record, quality_gates
    return None


def _read_record_evaluation(record: HistoryRecord) -> dict[str, Any]:
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    embedded = manifest.get("evaluation")
    if isinstance(embedded, dict) and embedded:
        return embedded
    sibling = record.path.parent / "evaluation.json"
    if not sibling.exists():
        status_file = _record_status_artifact(record, "evaluation.json")
        if status_file is None:
            return {}
        sibling = status_file
    try:
        with sibling.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_record_quality_gates(record: HistoryRecord) -> dict[str, Any]:
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    embedded = manifest.get("quality_gates")
    if isinstance(embedded, dict) and embedded:
        return embedded
    sibling = record.path.parent / "quality_gates.json"
    if not sibling.exists():
        status_file = _record_status_artifact(record, "quality_gates.json")
        if status_file is None:
            return {}
        sibling = status_file
    try:
        with sibling.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _record_status_artifact(record: HistoryRecord, filename: str) -> Path | None:
    parent = record.path.parent
    status_candidates = []
    if record.status:
        status = str(record.status)
        status_candidates.extend([status, status.replace("quarantined", "quarantine")])
    status_candidates.extend(["approved", "quarantine", "quarantined", "rejected", "failed"])
    for status in dict.fromkeys(status_candidates):
        candidate = parent / status / filename
        if candidate.exists():
            return candidate
    return None


def _quality_gate_rows_from_record(record: HistoryRecord | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    payload = _read_record_quality_gates(record)
    metrics = payload.get("metrics_checked") if isinstance(payload.get("metrics_checked"), dict) else {}
    failures = payload.get("failures") if isinstance(payload.get("failures"), list) else []
    if not metrics and not failures:
        return []
    failures_by_metric = {
        str(item.get("metric") or item.get("gate")): item
        for item in failures
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for metric, value in metrics.items():
        failure = failures_by_metric.get(str(metric))
        mandatory = _metric_is_mandatory(metric, failure)
        rows.append(
            {
                "id": str(metric),
                "metric": _humanize_gate_key(str(metric)),
                "observed": _sanitize_public_value(value),
                "operator": None if failure is None else "<=",
                "threshold": None if failure is None else _sanitize_public_value(failure.get("limit")),
                "mandatory": mandatory,
                "passed": failure is None,
                "status": "Aprovado" if failure is None else ("Reprovado" if mandatory else "Quarentena"),
                "source": "quality_gates.json",
                "evidence": _sanitize_public_value(value),
            }
        )
    for index, failure in enumerate(failures):
        if not isinstance(failure, dict):
            continue
        metric = str(failure.get("metric") or failure.get("gate") or f"failure_{index}")
        if metric in metrics:
            continue
        mandatory = bool(failure.get("mandatory"))
        rows.append(
            {
                "id": str(failure.get("gate") or metric),
                "metric": _humanize_gate_key(metric),
                "observed": _sanitize_public_value(failure.get("value")),
                "operator": "<=" if failure.get("limit") is not None else None,
                "threshold": _sanitize_public_value(failure.get("limit")),
                "mandatory": mandatory,
                "passed": False,
                "status": "Reprovado" if mandatory else "Quarentena",
                "source": "quality_gates.json",
                "evidence": _sanitize_public_value(failure),
            }
        )
    return rows


def _metric_is_mandatory(metric: Any, failure: Any) -> bool:
    if isinstance(failure, dict) and failure.get("mandatory") is not None:
        return bool(failure.get("mandatory"))
    name = str(metric)
    return name in {"invalid_rows", "null_required_fields", "duplicated_identifier", "exact_train_match_rate"}


def _resolve_config(config: Any) -> GovernanceSourceConfig:
    if isinstance(config, GovernanceSourceConfig):
        return config
    return GovernanceSourceConfig(
        artifacts_root=Path(getattr(config, "artifacts_root")),
        models_root=Path(getattr(config, "models_root")),
        audit_events_path=Path(getattr(config, "audit_events_path", Path("artifacts/web_audit/events.jsonl"))),
        default_model=str(getattr(config, "default_model", "programmatic")),
        approved_model_artifacts=getattr(config, "approved_model_artifacts", None),
    )


def _available_model_labels(approved_generation_artifacts: list[SavedModelArtifact]) -> str:
    available = {"programmatic"}
    available.update(artifact.model for artifact in approved_generation_artifacts)
    labels = {"programmatic": "Programático", "ctgan": "CTGAN", "simple_gan": "GAN simples"}
    return ", ".join(labels[model] for model in ("programmatic", "ctgan", "simple_gan") if model in available)


def _latest_artifact(artifacts: list[SavedModelArtifact]) -> SavedModelArtifact | None:
    if not artifacts:
        return None
    return sorted(artifacts, key=lambda item: item.created_at_utc or "", reverse=True)[0]


def _confirmation_result(artifact: SavedModelArtifact) -> str | None:
    metrics = artifact_quality_summary(artifact)
    approved = metrics.get("approved_confirmation_seeds")
    seeds = metrics.get("confirmation_seeds")
    if isinstance(approved, (int, float)) and isinstance(seeds, list):
        return f"{int(approved)}/{len(seeds)} seeds aprovadas"
    return None


def _artifact_limitations(artifact: SavedModelArtifact) -> list[str]:
    value = artifact.manifest.get("limitations")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _sanitized_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment") if isinstance(manifest.get("environment"), dict) else {}
    libraries = manifest.get("library_versions") or environment.get("library_versions") or {}
    python = _python_version_label(environment.get("python_version") or manifest.get("python_version"))
    ctgan = libraries.get("ctgan") or manifest.get("ctgan_version")
    tensorflow = libraries.get("tensorflow")
    torch = libraries.get("torch")
    pieces = [piece for piece in [python, f"CTGAN {ctgan}" if ctgan else None] if piece]
    return {
        "python": python,
        "ctgan": ctgan,
        "tensorflow": tensorflow,
        "torch": torch,
        "summary": "; ".join(pieces) if pieces else None,
    }


def _python_version_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parts = value.split()[0].split(".")
    if len(parts) >= 2:
        return f"Python {parts[0]}.{parts[1]}"
    return "Python"


def _library_label(manifest: dict[str, Any]) -> str | None:
    env = _sanitized_environment(manifest)
    if env.get("ctgan"):
        return f"ctgan {env['ctgan']}"
    return None


def _epochs(manifest: dict[str, Any]) -> int | None:
    for key in ("ctgan_config", "config"):
        config = manifest.get(key)
        if isinstance(config, dict) and config.get("epochs") is not None:
            try:
                return int(config["epochs"])
            except (TypeError, ValueError):
                return None
    return None


def _conditional_income_status(manifest: dict[str, Any]) -> str | None:
    aggregate = manifest.get("aggregate_metrics") if isinstance(manifest.get("aggregate_metrics"), dict) else {}
    if aggregate.get("renda_wasserstein_normalized_max") is not None:
        return "avaliado"
    return None


def _seed_values(approval: dict[str, Any], key: str) -> list[float]:
    by_seed = approval.get("by_seed") if isinstance(approval.get("by_seed"), dict) else {}
    values: list[float] = []
    for payload in by_seed.values():
        if isinstance(payload, dict) and payload.get(key) is not None:
            try:
                values.append(float(payload[key]))
            except (TypeError, ValueError):
                continue
    return values


def _row_values(rows: list[Any], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                continue
    return values


def _range_or_single(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values)}


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _section_status(indicators: list[dict[str, Any]]) -> str:
    return "Não avaliado" if not indicators or all(item.get("value") is None for item in indicators) else "Disponível"


def _metric(label: str, key: str, value: Any, source: str, help_text: str) -> dict[str, Any]:
    return {"label": label, "key": key, "value": value, "source": source, "help": help_text}


def _glossary(term: str, definition: str) -> dict[str, str]:
    return {"term": term, "definition": definition}


def _display_value(value: Any) -> Any:
    if isinstance(value, dict) and {"min", "max"} <= set(value):
        return f"{value['min']} a {value['max']}"
    return value if value is not None else NOT_EVALUATED


def _coverage_label(metrics: dict[str, Any]) -> str:
    geo_parts = []
    for label, key in [("estados", "state_coverage"), ("municípios", "municipality_coverage"), ("DDDs", "ddd_coverage"), ("Geo_Key", "geography_key_coverage")]:
        value = metrics.get(key)
        if value is not None:
            geo_parts.append(f"{label}: {value}")
    return "; ".join(geo_parts) if geo_parts else NOT_EVALUATED
