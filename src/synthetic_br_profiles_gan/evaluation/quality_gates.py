"""Quality gates configuráveis para aprovação de datasets gerados."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from synthetic_br_profiles_gan.config import ConfigDict


DEFAULT_QUALITY_GATES: ConfigDict = {
    "assessment_mode": "experimental",
    "min_evaluation_rows": 100,
    "invalid_rows_max": {"value": 0, "mandatory": True},
    "duplicated_identifier_max": {"value": 0, "mandatory": True},
    "null_required_fields_max": {"value": 0, "mandatory": True},
    "exact_train_match_rate_max": {"value": 0.01, "mandatory": True},
    "duplicate_base_row_rate_max": {"value": 0.01, "mandatory": False},
    "total_variation_distance_max": {"value": 0.25, "mandatory": False},
    "correlation_difference_max": {"value": 0.30, "mandatory": False},
}

QUALITY_GATE_DOCUMENTATION: dict[str, dict[str, Any]] = {
    "min_evaluation_rows": {
        "metric": "row_counts.synthetic",
        "interpretation": "Tamanho mínimo da amostra sintética antes de gates estatísticos sustentarem uma afirmação de aprovação.",
        "unit": "linhas",
        "limit": DEFAULT_QUALITY_GATES["min_evaluation_rows"],
        "reason": "Smoke runs pequenos validam a mecânica, não a qualidade estatística.",
        "missing_behavior": "falha obrigatória no modo approval; quarentena nos demais modos",
    },
    "invalid_rows_max": {
        "metric": "validation.invalid_rows",
        "interpretation": "Quantidade de linhas finais estruturalmente inválidas.",
        "unit": "linhas",
        "limit": 0,
        "reason": "Datasets com estrutura obrigatória inválida não devem ser aprovados.",
        "missing_behavior": "falha obrigatória",
    },
    "duplicated_identifier_max": {
        "metric": "validation.reason_counts.*_duplicado for identifiers",
        "interpretation": "Identificadores gerados duplicados.",
        "unit": "linhas",
        "limit": 0,
        "reason": "Colisões de identificadores tornam registros sintéticos inutilizáveis.",
        "missing_behavior": "falha obrigatória",
    },
    "null_required_fields_max": {
        "metric": "validation.reason_counts.null_required_fields",
        "interpretation": "Valores obrigatórios nulos.",
        "unit": "campos",
        "limit": 0,
        "reason": "Campos obrigatórios devem estar completos.",
        "missing_behavior": "falha obrigatória",
    },
    "exact_train_match_rate_max": {
        "metric": "evaluation.privacy.exact_train_match_rate",
        "interpretation": "Matches exatos com linhas de treinamento apenas sobre colunas de modelo.",
        "unit": "taxa",
        "limit": 0.01,
        "reason": "Taxa elevada de match exato pode indicar memorização.",
        "missing_behavior": "falha obrigatória",
    },
    "duplicate_base_row_rate_max": {
        "metric": "evaluation.privacy.duplicate_base_rows.duplicate_row_rate",
        "interpretation": "Duplicidade de combinações-base nas 11 colunas produzidas pelo modelo.",
        "unit": "taxa",
        "limit": 0.01,
        "reason": "Limite informativo inicial para acompanhar baixa diversidade e possível colapso de modo.",
        "missing_behavior": "não avaliado quando a métrica não existe",
    },
    "total_variation_distance_max": {
        "metric": "max holdout categorical total variation distance",
        "interpretation": "Maior diferença de distribuição categórica contra o holdout.",
        "unit": "distância em [0, 1]",
        "limit": 0.25,
        "reason": "Limite informativo padrão para desvio de distribuição.",
        "missing_behavior": "falha opcional quando configurada como opcional",
    },
    "correlation_difference_max": {
        "metric": "holdout correlation summary max_abs_difference",
        "interpretation": "Maior diferença absoluta entre matrizes de correlação.",
        "unit": "diferença absoluta de correlação",
        "limit": 0.30,
        "reason": "Limite informativo padrão para desvio nas relações.",
        "missing_behavior": "falha opcional quando configurada como opcional",
    },
}


@dataclass(frozen=True)
class QualityGateResult:
    """Status dos quality gates e motivos de falha."""

    status: str
    failures: list[dict[str, Any]]
    metrics_checked: dict[str, Any]


def _gate_spec(config: ConfigDict, key: str) -> dict[str, Any]:
    value = config.get(key, DEFAULT_QUALITY_GATES[key])
    if isinstance(value, dict):
        return {"value": value["value"], "mandatory": bool(value.get("mandatory", True))}
    return {"value": value, "mandatory": True}


def _is_valid_metric(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _metric_failure(gate_name: str, metric_name: str, current: Any, limit: Any, mandatory: bool, reason: str) -> dict[str, Any]:
    return {
        "gate": gate_name,
        "metric": metric_name,
        "value": current,
        "limit": limit,
        "mandatory": bool(mandatory),
        "reason": reason,
    }


def _max_tvd(evaluation: dict[str, Any]) -> float:
    categorical = evaluation.get("against_holdout", {}).get("categorical", {})
    values = [metric.get("total_variation_distance") for metric in categorical.values()]
    finite = [float(value) for value in values if _is_valid_metric(value)]
    return max(finite) if finite else float("nan")


def _correlation_difference(evaluation: dict[str, Any]) -> float:
    value = (
        evaluation.get("against_holdout", {})
        .get("correlations", {})
        .get("summary", {})
        .get("max_abs_difference")
    )
    return float(value) if _is_valid_metric(value) else float("nan")


def evaluate_quality_gates(
    validation: dict[str, Any],
    evaluation: dict[str, Any],
    config: ConfigDict | None = None,
) -> QualityGateResult:
    """Avalia quality gates e retorna approved, quarantined ou rejected."""
    gates = {**DEFAULT_QUALITY_GATES, **(config or {})}
    assessment_mode = str(gates.get("assessment_mode", "experimental"))
    reason_counts = validation.get("reason_counts", {})
    duplicated_identifier_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key.endswith("_duplicado") and key != "duplicated_rows"
    )
    privacy = evaluation.get("privacy", {}) if isinstance(evaluation.get("privacy"), dict) else {}
    duplicate_base = privacy.get("duplicate_base_rows") if isinstance(privacy.get("duplicate_base_rows"), dict) else {}
    exact_matches = privacy.get("exact_matches") if isinstance(privacy.get("exact_matches"), dict) else {}
    exact_train = exact_matches.get("train") if isinstance(exact_matches.get("train"), dict) else {}
    checked = {
        "synthetic_rows": evaluation.get("row_counts", {}).get("synthetic"),
        "invalid_rows": validation.get("invalid_rows"),
        "duplicated_identifier": float(duplicated_identifier_count),
        "null_required_fields": float(reason_counts.get("null_required_fields", 0)),
        "exact_train_match_rate": privacy.get("exact_train_match_rate"),
        "exact_train_match_count": exact_train.get("exact_match_count"),
        "duplicate_base_row_rate": duplicate_base.get("duplicate_row_rate") if duplicate_base else privacy.get("duplicate_row_rate"),
        "duplicate_base_duplicated_occurrences": duplicate_base.get("duplicated_occurrences"),
        "duplicate_base_duplicated_groups": duplicate_base.get("duplicated_groups"),
        "total_variation_distance": _max_tvd(evaluation),
        "correlation_difference": _correlation_difference(evaluation),
    }
    mapping = {
        "invalid_rows_max": ("invalid_rows", lambda current, limit: current <= limit),
        "duplicated_identifier_max": ("duplicated_identifier", lambda current, limit: current <= limit),
        "null_required_fields_max": ("null_required_fields", lambda current, limit: current <= limit),
        "exact_train_match_rate_max": ("exact_train_match_rate", lambda current, limit: current <= limit),
        "duplicate_base_row_rate_max": ("duplicate_base_row_rate", lambda current, limit: current <= limit),
        "total_variation_distance_max": ("total_variation_distance", lambda current, limit: current <= limit),
        "correlation_difference_max": ("correlation_difference", lambda current, limit: current <= limit),
    }

    failures: list[dict[str, Any]] = []
    min_rows = int(gates.get("min_evaluation_rows", DEFAULT_QUALITY_GATES["min_evaluation_rows"]))
    min_rows_mandatory = assessment_mode == "approval"
    synthetic_rows = checked["synthetic_rows"]
    if not _is_valid_metric(synthetic_rows) or float(synthetic_rows) < min_rows:
        failures.append(
            _metric_failure(
                "min_evaluation_rows",
                "synthetic_rows",
                synthetic_rows,
                min_rows,
                min_rows_mandatory,
                f"assessment_mode={assessment_mode}; smoke runs are technical checks, not approval evidence",
            )
        )

    for gate_name, (metric_name, predicate) in mapping.items():
        spec = _gate_spec(gates, gate_name)
        current = checked[metric_name]
        limit = float(spec["value"])
        if not _is_valid_metric(current):
            if gate_name == "duplicate_base_row_rate_max" and not bool(spec["mandatory"]):
                continue
            failures.append(
                _metric_failure(
                    gate_name,
                    metric_name,
                    current,
                    limit,
                    bool(spec["mandatory"]),
                    "metric_missing_or_invalid",
                )
            )
            continue
        current = float(current)
        if not predicate(current, limit):
            failures.append(_metric_failure(gate_name, metric_name, current, limit, bool(spec["mandatory"]), "threshold_failed"))

    if any(failure["mandatory"] for failure in failures):
        status = "rejected"
    elif failures:
        status = "quarantined"
    else:
        status = "approved"
    return QualityGateResult(status=status, failures=failures, metrics_checked=checked)
