"""Aprovação rastreável de artefatos neurais avaliados."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from synthetic_br_profiles_gan.domain.geography import GEOGRAPHY_MODEL_VERSION, GEO_KEY_COLUMN, geography_catalog_checksum
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, INCOME_MODEL_VERSION
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.registry import load_saved_synthesizer, load_training_manifest


APPROVAL_NOTE = (
    "A aprovação representa uma decisão interna baseada nos critérios técnicos do projeto. "
    "Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
)


class ApprovalValidationError(RuntimeError):
    """Falha de validação que bloqueia a aprovação de um artefato."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ApprovalPromotionResult:
    """Resultado da promoção de um artefato candidato para aprovado."""

    source_artifact: Path
    approved_artifact: Path
    approval_manifest_path: Path
    training_manifest_path: Path
    evidence_report: dict[str, Any]


def validate_ctgan_geo_v2_approval_evidence(
    artifact_path: str | Path,
    benchmark_path: str | Path,
    *,
    expected_seeds: Iterable[int] = (47, 48, 49),
) -> dict[str, Any]:
    """Valida as evidências obrigatórias antes da aprovação da CTGAN geography v2."""
    artifact = Path(artifact_path)
    benchmark = Path(benchmark_path)
    metadata = default_metadata()
    report: dict[str, Any] = {
        "artifact": str(artifact),
        "benchmark": str(benchmark),
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "mandatory_checks": {},
        "metrics_by_seed": {},
        "occupational_coverage": {},
        "blocked": False,
    }

    _check(report, "artifact_exists", artifact.exists() and artifact.is_dir(), str(artifact))
    _check(report, "benchmark_exists", benchmark.exists() and benchmark.is_dir(), str(benchmark))
    if not artifact.exists() or not benchmark.exists():
        return _raise_blocked(report, "Artefato ou benchmark de confirmação não encontrado.")

    manifest = load_training_manifest(artifact)
    _check(report, "model_is_ctgan", manifest.get("model") == "ctgan", manifest.get("model"))
    _check(report, "current_purpose_recommended_candidate", _normalized_status(manifest.get("purpose")) == "recommended_candidate", manifest.get("purpose"))
    _check(
        report,
        "current_status_recommended_candidate",
        _normalized_status(manifest.get("approval_status")) == "recommended_candidate",
        manifest.get("approval_status"),
    )
    _check(
        report,
        "categorical_vocabulary_version",
        int(manifest.get("categorical_vocabulary_version", 0)) == CATEGORICAL_VOCABULARY_VERSION,
        manifest.get("categorical_vocabulary_version"),
    )
    _check(
        report,
        "income_model_version",
        int(manifest.get("income_model_version", 0)) == INCOME_MODEL_VERSION,
        manifest.get("income_model_version"),
    )
    _check(
        report,
        "geography_model_version",
        int(manifest.get("geography_model_version", 0)) == GEOGRAPHY_MODEL_VERSION,
        manifest.get("geography_model_version"),
    )
    expected_checksum = geography_catalog_checksum()
    _check(
        report,
        "geography_catalog_checksum",
        manifest.get("geography_catalog_checksum") == expected_checksum,
        manifest.get("geography_catalog_checksum"),
    )
    _check(
        report,
        "external_schema_preserved",
        list(manifest.get("model_columns", [])) == metadata.model_columns and list(manifest.get("final_columns", [])) == metadata.final_columns,
        {"model_columns": manifest.get("model_columns"), "final_columns": manifest.get("final_columns")},
    )
    public_columns = set(manifest.get("model_columns", [])) | set(manifest.get("final_columns", []))
    _check(report, "geo_key_absent_from_public_schema", GEO_KEY_COLUMN not in public_columns, sorted(public_columns))

    try:
        loaded = load_saved_synthesizer(artifact, expected_model="ctgan")
        synthesizer = loaded.synthesizer
        loaded_ok = getattr(synthesizer, "geography_model_version", None) == GEOGRAPHY_MODEL_VERSION
        loaded_checksum = getattr(synthesizer, "geography_catalog_checksum", None)
        _check(report, "model_loadable", loaded_ok and loaded_checksum == expected_checksum, {"geography_model_version": getattr(synthesizer, "geography_model_version", None), "checksum": loaded_checksum})
    except Exception as exc:
        _check(report, "model_loadable", False, f"{type(exc).__name__}: {exc}")

    run_summary = _read_csv_dicts(benchmark / "run_summary.csv")
    results = _read_csv_dicts(benchmark / "results.csv")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    summary_by_seed = {int(row.get("seed", -1)): row for row in run_summary if row.get("model") == "ctgan"}
    _check(report, "three_seeds_present", set(summary_by_seed) >= expected_seed_set, sorted(summary_by_seed))

    for seed in sorted(expected_seed_set):
        row = summary_by_seed.get(seed)
        if row is None:
            report["metrics_by_seed"][str(seed)] = {"missing": True}
            continue
        raw_global = _derived_raw_global_validity(row)
        metrics = {
            "status": row.get("status"),
            "invalid_rows": _float(row.get("invalid_rows")),
            "valid_rows": _float(row.get("valid_rows")),
            "duplicated_identifiers": _metric_value(results, seed, "validation", "duplicated_identifiers"),
            "duplicate_base_row_rate": _float(row.get("duplicate_base_row_rate")),
            "duplicate_base_duplicated_occurrences": _float(row.get("duplicate_base_duplicated_occurrences")),
            "exact_train_match_rate": _float(row.get("exact_train_match_rate")),
            "exact_train_match_count": _float(row.get("exact_train_match_count")),
            "known_geography_key_rate": _float(row.get("known_geography_key_rate_raw")),
            "raw_geographic_validity_rate": _float(row.get("raw_geographic_validity_rate")),
            "raw_global_validity_rate": raw_global,
            "state_coverage": _float(row.get("state_coverage")),
            "municipality_coverage": _float(row.get("municipality_coverage")),
            "ddd_coverage": _float(row.get("ddd_coverage")),
            "geography_key_coverage": _float(row.get("geography_key_coverage_raw")),
            "geography_key_distribution_tvd": _float(row.get("geography_key_distribution_tvd_raw")),
            "peak_memory_mb": _float(row.get("peak_memory_mb")),
            "training_seconds": _float(row.get("training_seconds")),
            "generation_seconds": _float(row.get("generation_seconds")),
        }
        report["metrics_by_seed"][str(seed)] = metrics
        _check(report, f"seed_{seed}_approved", metrics["status"] == "approved", metrics["status"])
        _check(report, f"seed_{seed}_zero_invalid_rows", metrics["invalid_rows"] == 0.0, metrics["invalid_rows"])
        _check(report, f"seed_{seed}_zero_duplicated_identifiers", metrics["duplicated_identifiers"] == 0.0, metrics["duplicated_identifiers"])
        _check(report, f"seed_{seed}_zero_duplicate_base", metrics["duplicate_base_row_rate"] == 0.0 and metrics["duplicate_base_duplicated_occurrences"] == 0.0, metrics)
        _check(report, f"seed_{seed}_zero_exact_train_match", metrics["exact_train_match_rate"] == 0.0 and metrics["exact_train_match_count"] == 0.0, metrics)
        _check(report, f"seed_{seed}_known_geography_key_rate", metrics["known_geography_key_rate"] == 1.0, metrics["known_geography_key_rate"])
        _check(report, f"seed_{seed}_raw_geographic_validity", metrics["raw_geographic_validity_rate"] == 1.0, metrics["raw_geographic_validity_rate"])
        _check(report, f"seed_{seed}_raw_global_validity_range", raw_global is not None and 0.915 <= raw_global <= 0.969, raw_global)
        _check(report, f"seed_{seed}_state_coverage", metrics["state_coverage"] == 1.0, metrics["state_coverage"])
        _check(report, f"seed_{seed}_municipality_coverage", metrics["municipality_coverage"] == 1.0, metrics["municipality_coverage"])
        _check(report, f"seed_{seed}_ddd_coverage", metrics["ddd_coverage"] == 1.0, metrics["ddd_coverage"])
        _check(report, f"seed_{seed}_geography_key_coverage", metrics["geography_key_coverage"] == 1.0, metrics["geography_key_coverage"])

    report["occupational_coverage"] = _occupation_coverage(results)
    _check(
        report,
        "occupational_coverage_documented",
        report["occupational_coverage"].get("47", {}).get("coverage_count") == "37/37"
        and report["occupational_coverage"].get("49", {}).get("coverage_count") == "37/37"
        and report["occupational_coverage"].get("48", {}).get("coverage_count") == "36/37"
        and report["occupational_coverage"].get("48", {}).get("missing_occupations") == ["Diretor"],
        report["occupational_coverage"],
    )

    report["library_versions"] = _read_json(benchmark / "environment.json").get("library_versions", {})
    report["summary"] = {
        "approved_runs": sum(1 for seed in expected_seed_set if report["metrics_by_seed"].get(str(seed), {}).get("status") == "approved"),
        "raw_global_validity_min": min(_present(metric.get("raw_global_validity_rate") for metric in report["metrics_by_seed"].values())),
        "raw_global_validity_max": max(_present(metric.get("raw_global_validity_rate") for metric in report["metrics_by_seed"].values())),
        "geography_catalog_checksum": expected_checksum,
        "geography_model_version": GEOGRAPHY_MODEL_VERSION,
        "income_model_version": INCOME_MODEL_VERSION,
        "categorical_vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
    }

    failures = [name for name, check in report["mandatory_checks"].items() if not check["passed"]]
    if failures:
        return _raise_blocked(report, "Evidências obrigatórias ausentes ou inválidas: " + ", ".join(failures))
    return report


def promote_ctgan_geo_v2_artifact(
    source_artifact: str | Path,
    benchmark_path: str | Path,
    destination_root: str | Path,
    *,
    approved_at_utc: datetime | None = None,
) -> ApprovalPromotionResult:
    """Cria uma cópia aprovada do artefato CTGAN geography v2 depois da validação."""
    source = Path(source_artifact)
    benchmark = Path(benchmark_path)
    evidence = validate_ctgan_geo_v2_approval_evidence(source, benchmark)
    timestamp = (approved_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(destination_root) / f"{timestamp}-income-v3-geo-v2-approved"
    if destination.exists():
        raise FileExistsError(f"Approved artifact already exists: {destination}")
    shutil.copytree(source, destination)

    source_manifest = load_training_manifest(source)
    backup_path = destination / "manifest.before-approval.json"
    _write_json(source_manifest, backup_path)
    _write_json(source_manifest, destination / "training_manifest.before-approval.json")

    approved_instant = (approved_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    approval_manifest = _approval_manifest(
        source_artifact=source,
        approved_artifact=destination,
        source_manifest=source_manifest,
        evidence=evidence,
        approved_at_utc=approved_instant,
    )
    approval_path = destination / "approval_manifest.json"
    _write_json(approval_manifest, approval_path)

    training_manifest = dict(source_manifest)
    training_manifest.update(
        {
            "purpose": "approved",
            "approval_status": "approved",
            "approved_at_utc": approved_instant,
            "decision_type": "technical_internal_approval",
            "recommended_for_neural_generation": True,
            "general_platform_default": False,
            "source_artifact": _artifact_id(source),
            "approval_manifest": "approval_manifest.json",
            "approval_note": APPROVAL_NOTE,
            "approval_evidence_summary": approval_manifest["metrics_summary"],
        }
    )
    _write_json(training_manifest, destination / "training_manifest.json")
    return ApprovalPromotionResult(
        source_artifact=source,
        approved_artifact=destination,
        approval_manifest_path=approval_path,
        training_manifest_path=destination / "training_manifest.json",
        evidence_report=evidence,
    )


def _approval_manifest(
    *,
    source_artifact: Path,
    approved_artifact: Path,
    source_manifest: dict[str, Any],
    evidence: dict[str, Any],
    approved_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "model_approval",
        "artifact_id": _artifact_id(approved_artifact),
        "source_artifact_id": _artifact_id(source_artifact),
        "previous_status": source_manifest.get("approval_status") or source_manifest.get("purpose"),
        "new_status": "approved",
        "approved_at_utc": approved_at_utc,
        "decision_type": "technical_internal_approval",
        "evidence_benchmarks": [evidence["benchmark"]],
        "confirmation_seeds": evidence["expected_seeds"],
        "mandatory_checks": evidence["mandatory_checks"],
        "metrics_summary": evidence["summary"] | {"by_seed": evidence["metrics_by_seed"]},
        "known_limitations": [
            "A seed 48 cobriu 36/37 ocupações; a ocupação ausente foi Diretor, categoria rara.",
            "A fonte local associa DDDs ao estado, não a um DDD oficial único por município.",
            "A aprovação é técnica e interna; não constitui certificação externa, garantia de anonimização ou validação populacional oficial.",
            "O modelo programático permanece como padrão geral da plataforma.",
        ],
        "library_versions": evidence.get("library_versions", {}),
        "vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "income_model_version": INCOME_MODEL_VERSION,
        "geography_model_version": GEOGRAPHY_MODEL_VERSION,
        "geography_catalog_checksum": geography_catalog_checksum(),
        "recommended_for_neural_generation": True,
        "general_platform_default": False,
        "approval_note": APPROVAL_NOTE,
    }


def _check(report: dict[str, Any], name: str, passed: bool, value: Any) -> None:
    report["mandatory_checks"][name] = {"passed": bool(passed), "value": value}


def _raise_blocked(report: dict[str, Any], message: str) -> dict[str, Any]:
    report["blocked"] = True
    raise ApprovalValidationError(message, report)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ApprovalValidationError(f"Arquivo de evidência ausente: {path}", {"missing_file": str(path), "blocked": True})
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(rows: list[dict[str, str]], seed: int, metric_group: str, metric_name: str) -> float | None:
    for row in rows:
        if int(row.get("seed", -1)) == int(seed) and row.get("metric_group") == metric_group and row.get("metric_name") == metric_name:
            return _float(row.get("value"))
    return None


def _derived_raw_global_validity(row: dict[str, str]) -> float | None:
    explicit = _float(row.get("raw_structural_validity_rate"))
    if explicit is not None:
        return explicit
    values = [
        _float(row.get("raw_geographic_validity_rate")),
        _float(row.get("raw_professional_validity_rate")),
        _float(row.get("raw_non_relational_validity_rate")),
    ]
    present = _present(values)
    return min(present) if present else None


def _occupation_coverage(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_seed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("metric_group") != "categorical" or row.get("column") != "Ocupacao":
            continue
        seed = str(row.get("seed"))
        payload = by_seed.setdefault(seed, {})
        if row.get("metric_name") == "category_coverage_holdout":
            coverage = _float(row.get("value"))
            payload["coverage_rate"] = coverage
            if coverage is not None:
                payload["coverage_count"] = f"{round(coverage * 37):.0f}/37"
        elif row.get("metric_name") == "missing_categories_count":
            try:
                payload["missing_occupations"] = json.loads(row.get("details") or "[]")
            except json.JSONDecodeError:
                payload["missing_occupations"] = []
    return by_seed


def _present(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _normalized_status(value: Any) -> str:
    return str(value or "").lower().replace("-", "_")


def _artifact_id(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "ctgan":
        return f"ctgan/{path.name}"
    return path.name
