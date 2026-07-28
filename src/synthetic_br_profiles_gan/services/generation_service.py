"""Serviço reutilizável de geração a partir de sintetizadores salvos ou programáticos."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.column_catalog import ColumnSelection, resolve_column_selection
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge
from synthetic_br_profiles_gan.exceptions import ConfigurationError, StructuralValidationError
from synthetic_br_profiles_gan.manifest import environment_info, get_git_commit, write_json
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.models.registry import load_saved_synthesizer
from synthetic_br_profiles_gan.pipeline import DEFAULT_PIPELINE_CONFIG, generate_profiles
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


SUPPORTED_OUTPUT_FORMATS = {"csv", "json", "parquet"}
GOVERNANCE_NOTICE = (
    "Os dados deste arquivo são sintéticos e não foram consultados ou validados em bases oficiais. "
    "A validade estrutural de um documento não comprova sua existência, regularidade ou associação a uma pessoa real."
)


@dataclass(frozen=True)
class GenerationRequest:
    """Solicitação de geração independente da CLI."""

    model: str | None
    model_path: Path | None
    num_rows: int
    output_path: Path
    output_format: str
    seed: int = 41
    config: dict[str, Any] | None = None
    overwrite: bool = False
    selected_columns: tuple[str, ...] | list[str] | None = None
    column_preset: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Resultado estruturado de uma geração exportada."""

    model: str
    num_rows: int
    output_path: Path
    manifest_path: Path
    duration_seconds: float
    validation_report: dict[str, Any]
    internal_columns: tuple[str, ...]
    exported_columns: tuple[str, ...]


def run_generation(request: GenerationRequest) -> GenerationResult:
    """Gera, valida, exporta e cria manifesto de um dataset sintético."""
    started = datetime.now(timezone.utc)
    perf_start = time.perf_counter()
    model_name = _normalize_model_name(request.model) if request.model else None
    output_format = str(request.output_format).lower()
    _validate_generation_request(request, model_name, output_format)
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, request.config or {})
    metadata = default_metadata()
    column_selection = resolve_column_selection(
        request.selected_columns,
        preset=request.column_preset,
        available_columns=metadata.final_columns,
    )
    reference_date = str(config.get("reference_date", DEFAULT_PIPELINE_CONFIG["reference_date"]))
    output_path = Path(request.output_path)
    manifest_path = _generation_manifest_path(output_path)

    load_started = time.perf_counter()
    model_artifact: str | None = None
    training_manifest: dict[str, Any] | None = None
    if request.model_path is not None:
        loaded = load_saved_synthesizer(request.model_path, expected_model=model_name)
        synthesizer = loaded.synthesizer
        model_name = loaded.model
        model_artifact = str(loaded.artifact_path)
        training_manifest = loaded.manifest
    else:
        if model_name != "programmatic":
            raise ConfigurationError("--model-path is required for ctgan and simple_gan generation.")
        model_config = deep_merge(config.get("models", {}).get("programmatic", {}), {"seed": int(request.seed) + 100_003})
        synthesizer = ProgrammaticSynthesizer(model_config)
        synthesizer.fit(pd.DataFrame(columns=metadata.model_columns), metadata)
    loading_seconds = float(time.perf_counter() - load_started)

    seed_tensorflow = model_name == "simple_gan"
    seed_torch = model_name == "ctgan"
    set_global_seed(int(request.seed), seed_tensorflow=seed_tensorflow, seed_torch=seed_torch)
    _set_synthesizer_seed(synthesizer, int(request.seed), model_name or "")

    generation_config = dict(config.get("generation", {}))
    batch_size = max(int(generation_config.get("batch_size", 1024)), int(request.num_rows))
    max_batches = max(int(generation_config.get("max_batches", 20)), 1)
    sampling_started = time.perf_counter()
    dataset, accounting, candidate_validation = generate_profiles(
        synthesizer=synthesizer,
        n_target=int(request.num_rows),
        metadata=metadata,
        seed=int(request.seed),
        reference_date=reference_date,
        batch_size=batch_size,
        max_batches=max_batches,
        date_format=str(generation_config.get("date_format", "%Y-%m-%d")),
    )
    generation_seconds = float(time.perf_counter() - sampling_started)
    sampling_seconds = float(accounting.get("sampling_seconds", generation_seconds))
    postprocessing_seconds = float(accounting.get("postprocessing_seconds", 0.0))

    validation_started = time.perf_counter()
    validation_report = validate_profile_dataframe(dataset, metadata=metadata, final=True, reference_date=reference_date).report
    validation_seconds = float(time.perf_counter() - validation_started)
    validation_report = {
        **validation_report,
        "validation_scope": "full_final_schema",
        "validated_columns": list(metadata.final_columns),
        "projection_after_validation": True,
    }
    if int(len(dataset)) != int(request.num_rows):
        raise StructuralValidationError(f"Generated dataset has {len(dataset)} rows; expected {request.num_rows}.")
    if list(dataset.columns) != metadata.final_columns:
        raise StructuralValidationError("Generated dataset columns do not match the canonical final schema.")
    if not validation_report.get("is_valid", False):
        raise StructuralValidationError(f"Generated dataset failed structural validation: {validation_report.get('reason_counts', {})}")

    exported_dataset = dataset.loc[:, list(column_selection.exported_columns)].copy()
    export_started = time.perf_counter()
    _export_dataset(exported_dataset, output_path, output_format)
    export_seconds = float(time.perf_counter() - export_started)
    ended = datetime.now(timezone.utc)
    manifest = build_generation_manifest(
        model=model_name or "programmatic",
        model_artifact=model_artifact,
        rows=int(request.num_rows),
        columns=list(exported_dataset.columns),
        internal_columns=list(dataset.columns),
        column_selection=column_selection,
        output_format=output_format,
        seed=int(request.seed),
        output_path=output_path,
        validation_report=validation_report,
        timings={
            "model_loading_seconds": loading_seconds,
            "sampling_seconds": sampling_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "validation_seconds": validation_seconds,
            "export_seconds": export_seconds,
            "total_seconds": float(time.perf_counter() - perf_start),
        },
        started_at=started,
        ended_at=ended,
        training_manifest=training_manifest,
        candidate_validation=candidate_validation,
        generation_accounting=accounting,
    )
    write_json(manifest, manifest_path)
    return GenerationResult(
        model=model_name or "programmatic",
        num_rows=int(request.num_rows),
        output_path=output_path,
        manifest_path=manifest_path,
        duration_seconds=float((ended - started).total_seconds()),
        validation_report=validation_report,
        internal_columns=tuple(dataset.columns),
        exported_columns=tuple(exported_dataset.columns),
    )


def build_generation_manifest(
    model: str,
    model_artifact: str | None,
    rows: int,
    columns: list[str],
    internal_columns: list[str],
    column_selection: ColumnSelection,
    output_format: str,
    seed: int,
    output_path: Path,
    validation_report: dict[str, Any],
    timings: dict[str, float],
    started_at: datetime,
    ended_at: datetime,
    training_manifest: dict[str, Any] | None,
    candidate_validation: dict[str, Any],
    generation_accounting: dict[str, Any],
) -> dict[str, Any]:
    """Cria o manifesto de uma geração exportada."""
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_dataset",
        "created_at_utc": started_at.astimezone(timezone.utc).isoformat(),
        "ended_at_utc": ended_at.astimezone(timezone.utc).isoformat(),
        "model": model,
        "model_artifact": model_artifact,
        "source_training_manifest": None if training_manifest is None else training_manifest.get("created_at_utc"),
        "rows": int(rows),
        "columns": columns,
        "requested_columns": (
            None if column_selection.requested_columns is None else list(column_selection.requested_columns)
        ),
        "exported_columns": list(column_selection.exported_columns),
        "internally_generated_columns": internal_columns,
        "column_selection_mode": column_selection.mode,
        "column_preset": column_selection.preset,
        "internal_dependencies": {
            column: list(dependencies)
            for column, dependencies in column_selection.internal_dependencies.items()
        },
        "format": output_format,
        "seed": int(seed),
        "output_file": str(output_path),
        "output_size_bytes": int(output_path.stat().st_size),
        "timings": timings,
        "validation": validation_report,
        "candidate_validation": candidate_validation,
        "generation_accounting": generation_accounting,
        "environment": environment_info(),
        "git_commit": get_git_commit(Path.cwd()),
        "reproducibility": {
            "determinism_scope": (
                "Determinístico para o gerador programático e para pós-processamento com a mesma seed. "
                "Backends neurais podem ter limitações de determinismo dependendo da biblioteca e do hardware."
            )
        },
        "governance_notice": GOVERNANCE_NOTICE,
    }


def _validate_generation_request(request: GenerationRequest, model: str | None, output_format: str) -> None:
    if model is not None and model not in {"programmatic", "simple_gan", "ctgan"}:
        raise ConfigurationError(f"Unknown generation model: {request.model}")
    if request.model_path is None and model is None:
        raise ConfigurationError("Either --model-path or --model must be provided.")
    if int(request.num_rows) <= 0:
        raise ConfigurationError("rows must be greater than zero.")
    if int(request.seed) < 0:
        raise ConfigurationError("seed must be a non-negative integer.")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ConfigurationError(f"Unsupported output format: {request.output_format}")
    output_path = Path(request.output_path)
    manifest_path = _generation_manifest_path(output_path)
    existing = [path for path in [output_path, manifest_path] if path.exists()]
    if existing and not request.overwrite:
        raise ConfigurationError(f"Output file already exists and overwrite is false: {existing[0]}")
    if request.model_path is not None and not Path(request.model_path).exists():
        raise ConfigurationError(f"Model artifact path does not exist: {request.model_path}")


def _export_dataset(dataset: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        dataset.to_csv(output_path, index=False, encoding="utf-8", sep=";")
        return
    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(dataset.to_dict(orient="records"), file, ensure_ascii=False, indent=2, default=str)
        return
    if output_format == "parquet":
        dataset.to_parquet(output_path, index=False)
        return
    raise ConfigurationError(f"Unsupported output format: {output_format}")


def _generation_manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".manifest.json")


def _normalize_model_name(model: str | None) -> str | None:
    if model is None:
        return None
    normalized = model.lower().replace("-", "_")
    aliases = {
        "simple_tabular_gan": "simple_gan",
        "dense_tabular_gan": "simple_gan",
        "ctgan_synthesizer": "ctgan",
        "programmatic_synthesizer": "programmatic",
    }
    return aliases.get(normalized, normalized)


def _set_synthesizer_seed(synthesizer: Any, seed: int, model: str) -> None:
    if hasattr(synthesizer, "config") and isinstance(synthesizer.config, dict):
        model_seed = int(seed) + 100_003 if model == "programmatic" else int(seed)
        synthesizer.config = {**synthesizer.config, "seed": model_seed}
    if hasattr(synthesizer, "_sample_calls"):
        synthesizer._sample_calls = 0
