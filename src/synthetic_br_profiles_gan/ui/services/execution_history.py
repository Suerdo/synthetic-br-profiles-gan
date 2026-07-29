"""Leitura conservadora de manifestos para histórico da interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class HistoryRecord:
    """Registro resumido de uma execução, benchmark ou artefato de modelo."""

    kind: str
    identifier: str
    path: Path
    created_at_utc: str | None
    model: str | None
    seed: int | None
    status: str | None
    rows: int | None
    train_rows: int | None
    duration_seconds: float | None
    vocabulary_version: int | None
    summary: str
    manifest: dict[str, Any]


def load_history(artifacts_root: str | Path, limit: int | None = None) -> list[HistoryRecord]:
    """Carrega histórico resumido a partir dos manifestos existentes."""
    root = Path(artifacts_root)
    if not root.exists():
        return []
    records: list[HistoryRecord] = []
    records.extend(_load_run_records(root / "runs"))
    records.extend(_load_benchmark_records(root / "benchmarks"))
    records.extend(_load_model_records(root / "models"))
    records.extend(_load_generation_records(root / "ui_sessions"))
    records.sort(key=lambda item: item.created_at_utc or "", reverse=True)
    if limit is not None:
        return records[: int(limit)]
    return records


def filter_history(
    records: Iterable[HistoryRecord],
    *,
    kind: str | None = None,
    model: str | None = None,
    status: str | None = None,
    seed: int | None = None,
) -> list[HistoryRecord]:
    """Filtra registros do histórico por campos de alto nível."""
    filtered: list[HistoryRecord] = []
    for record in records:
        if kind and record.kind != kind:
            continue
        if model and record.model != model:
            continue
        if status and record.status != status:
            continue
        if seed is not None and record.seed != seed:
            continue
        filtered.append(record)
    return filtered


def history_summary(records: Iterable[HistoryRecord]) -> dict[str, Any]:
    """Consolida indicadores simples do histórico disponível."""
    materialized = list(records)
    latest = materialized[0] if materialized else None
    status_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for record in materialized:
        status_counts[str(record.status or "Não disponível")] = status_counts.get(str(record.status or "Não disponível"), 0) + 1
        if record.model:
            model_counts[record.model] = model_counts.get(record.model, 0) + 1
    return {
        "total_records": len(materialized),
        "latest_identifier": None if latest is None else latest.identifier,
        "latest_status": None if latest is None else latest.status,
        "latest_kind": None if latest is None else latest.kind,
        "status_counts": status_counts,
        "model_counts": model_counts,
    }


def history_as_rows(records: Iterable[HistoryRecord]) -> list[dict[str, Any]]:
    """Converte registros em linhas seguras para exibição tabular."""
    rows: list[dict[str, Any]] = []
    for record in records:
        privacy = _record_privacy(record)
        duplicate_base = privacy.get("duplicate_base_rows") if isinstance(privacy.get("duplicate_base_rows"), dict) else {}
        rows.append(
            {
                "tipo": record.kind,
                "identificador": record.identifier,
                "modelo": record.model or "Não disponível",
                "seed": record.seed if record.seed is not None else "Não disponível",
                "status": record.status or "Não disponível",
                "linhas": record.rows if record.rows is not None else "Não disponível",
                "treino": record.train_rows if record.train_rows is not None else "Não disponível",
                "duração_s": record.duration_seconds if record.duration_seconds is not None else "Não disponível",
                "vocabulário": record.vocabulary_version if record.vocabulary_version is not None else "Não disponível",
                "duplicidade_base": _not_evaluated_if_none(duplicate_base.get("duplicate_row_rate")),
                "match_exato_treino": _not_evaluated_if_none(privacy.get("exact_train_match_rate")),
                "combinações_únicas": _not_evaluated_if_none(privacy.get("unique_combinations")),
                "criado_em_utc": record.created_at_utc or "Não disponível",
                "resumo": record.summary,
            }
        )
    return rows


def _load_run_records(root: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    for manifest_path in _iter_json_files(root, "manifest.json"):
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        identifier = str(manifest.get("run_id") or manifest_path.parent.name)
        records.append(
            HistoryRecord(
                kind="pipeline_run",
                identifier=identifier,
                path=manifest_path,
                created_at_utc=manifest.get("timestamp_utc") or manifest.get("created_at_utc"),
                model=manifest.get("model"),
                seed=_optional_int(manifest.get("seed")),
                status=manifest.get("status"),
                rows=_optional_int(manifest.get("generated_rows") or manifest.get("requested_rows")),
                train_rows=_optional_int(manifest.get("train_rows")),
                duration_seconds=_optional_float(manifest.get("duration_seconds")),
                vocabulary_version=_optional_int(manifest.get("categorical_vocabulary_version")),
                summary="Execução de pipeline registrada por manifesto.",
                manifest=manifest,
            )
        )
    return records


def _load_benchmark_records(root: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    for manifest_path in _iter_json_files(root, "benchmark_manifest.json"):
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        identifier = str(manifest.get("benchmark_id") or manifest_path.parent.name)
        records.append(
            HistoryRecord(
                kind="benchmark",
                identifier=identifier,
                path=manifest_path,
                created_at_utc=manifest.get("timestamp_utc") or manifest.get("created_at_utc"),
                model=",".join(manifest.get("models", [])) if isinstance(manifest.get("models"), list) else None,
                seed=None,
                status=manifest.get("status"),
                rows=_optional_int(manifest.get("expected_runs")),
                train_rows=None,
                duration_seconds=_optional_float(manifest.get("duration_seconds")),
                vocabulary_version=None,
                summary="Benchmark registrado por manifesto.",
                manifest=manifest,
            )
        )
    return records


def _load_model_records(root: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    for manifest_path in _iter_json_files(root, "training_manifest.json"):
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        records.append(
            HistoryRecord(
                kind="model_training",
                identifier=manifest_path.parent.name,
                path=manifest_path,
                created_at_utc=manifest.get("created_at_utc"),
                model=manifest.get("model"),
                seed=_optional_int(manifest.get("seed")),
                status=manifest.get("approval_status") or manifest.get("purpose") or "Não avaliado",
                rows=_optional_int(manifest.get("calibration_rows")),
                train_rows=_optional_int(manifest.get("train_rows")),
                duration_seconds=_optional_float(manifest.get("timings", {}).get("total_seconds") if isinstance(manifest.get("timings"), dict) else manifest.get("duration_seconds")),
                vocabulary_version=_optional_int(manifest.get("categorical_vocabulary_version")),
                summary="Artefato de sintetizador treinado.",
                manifest=manifest,
            )
        )
    return records


def _load_generation_records(root: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    for manifest_path in _iter_json_files(root, "*.manifest.json"):
        manifest = _read_json(manifest_path)
        if not manifest or manifest.get("artifact_type") != "synthetic_dataset":
            continue
        records.append(
            HistoryRecord(
                kind="ui_generation",
                identifier=manifest_path.stem,
                path=manifest_path,
                created_at_utc=manifest.get("created_at_utc"),
                model=manifest.get("model"),
                seed=_optional_int(manifest.get("seed")),
                status="approved" if manifest.get("validation", {}).get("is_valid") else "rejected",
                rows=_optional_int(manifest.get("rows")),
                train_rows=None,
                duration_seconds=_optional_float(manifest.get("timings", {}).get("total_seconds") if isinstance(manifest.get("timings"), dict) else None),
                vocabulary_version=_optional_int(manifest.get("output_vocabulary_version")),
                summary="Geração realizada pela interface.",
                manifest=manifest,
            )
        )
    return records


def _iter_json_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            content = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return content if isinstance(content, dict) else {}


def _record_privacy(record: HistoryRecord) -> dict[str, Any]:
    manifest = record.manifest if isinstance(record.manifest, dict) else {}
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = _read_json(record.path.parent / "evaluation.json")
    privacy = evaluation.get("privacy") if isinstance(evaluation, dict) else None
    return privacy if isinstance(privacy, dict) else {}


def _not_evaluated_if_none(value: Any) -> Any:
    return "Não avaliado" if value is None else value


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
