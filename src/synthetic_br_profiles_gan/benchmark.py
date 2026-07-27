"""Benchmark orchestration for comparing tabular synthesizers."""

from __future__ import annotations

import logging
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.calibration import (
    DEFAULT_CALIBRATION_CONFIG,
    generate_calibration_dataset,
    save_calibration_splits,
    split_train_holdout,
)
from synthetic_br_profiles_gan.config import (
    ConfigDict,
    config_hash,
    deep_merge,
    save_yaml_config,
    validate_benchmark_config,
)
from synthetic_br_profiles_gan.evaluation.quality_gates import DEFAULT_QUALITY_GATES
from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable, PipelineError
from synthetic_br_profiles_gan.manifest import (
    build_run_id,
    environment_info,
    get_git_commit,
    hash_file,
    write_json,
)
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.pipeline import DEFAULT_PIPELINE_CONFIG, run_pipeline_on_splits

LOGGER = logging.getLogger(__name__)

DEFAULT_BENCHMARK_CONFIG: ConfigDict = {
    "benchmark": {
        "name": "pilot",
        "models": ["programmatic", "simple_gan", "ctgan"],
        "seeds": [11, 22, 33],
        "calibration_rows": 5000,
        "synthetic_rows": 2000,
        "holdout_fraction": 0.20,
        "assessment_mode": "experimental",
        "continue_on_error": True,
        "reference_date": "2026-07-26",
    },
    "calibration": {
        "income": DEFAULT_CALIBRATION_CONFIG["income"],
        "age": DEFAULT_CALIBRATION_CONFIG["age"],
        "region_weights": DEFAULT_CALIBRATION_CONFIG["region_weights"],
    },
    "models": {
        "programmatic": {},
        "simple_gan": {
            "epochs": 20,
            "batch_size": 128,
            "latent_dim": 32,
            "verbose_every": 0,
            "metrics_every": 0,
        },
        "ctgan": {
            "epochs": 20,
            "batch_size": 500,
            "verbose": False,
            "enable_gpu": False,
            "cuda": None,
        },
    },
    "generation": {
        "batch_size": 2048,
        "max_batches": 20,
        "date_format": "%Y-%m-%d",
    },
    "evaluation": DEFAULT_PIPELINE_CONFIG["evaluation"],
    "quality_gates": DEFAULT_QUALITY_GATES,
    "outputs": {
        "base_directory": "artifacts/benchmarks",
        "export_csv": True,
        "export_parquet": True,
        "export_json": True,
        "export_individual_xlsx": False,
    },
    "execution": {"parallelism": 1},
    "ranking": {"enabled": False},
}

NUMERIC_COLUMNS = ["Idade", "Renda", "Dependentes"]
CATEGORICAL_COLUMNS = ["Genero", "Regiao", "Estado", "Escolaridade", "Estado_Civil", "Ocupacao"]
MAIN_SUMMARY_METRICS = [
    "renda_wasserstein_normalized",
    "renda_ks",
    "genero_tvd",
    "correlation_difference",
    "duplicate_row_rate",
    "exact_train_match_rate",
    "dcr",
    "nndr",
    "training_seconds",
    "generation_seconds",
]
LONG_RESULT_COLUMNS = [
    "benchmark_id",
    "run_id",
    "model",
    "seed",
    "status",
    "metric_group",
    "metric_name",
    "column",
    "value",
    "reference",
    "difference",
    "details",
]
RUN_SUMMARY_COLUMNS = [
    "benchmark_id",
    "run_id",
    "model",
    "seed",
    "status",
    "duration_seconds",
    "calibration_seconds",
    "split_seconds",
    "training_seconds",
    "generation_seconds",
    "postprocessing_seconds",
    "validation_seconds",
    "evaluation_seconds",
    "export_seconds",
    "requested_rows",
    "generated_rows",
    "valid_rows",
    "invalid_rows",
    "renda_wasserstein_normalized",
    "renda_wasserstein",
    "renda_ks",
    "genero_tvd",
    "correlation_difference",
    "duplicate_row_rate",
    "unique_combination_rate",
    "exact_train_match_rate",
    "exact_holdout_match_rate",
    "dcr",
    "nndr",
]


def build_benchmark_id(name: str, timestamp: datetime | None = None) -> str:
    """Build a stable benchmark id prefix plus a timestamp/short suffix."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip().lower()).strip("-") or "benchmark"
    return f"{slug}-{build_run_id(timestamp)}"


def resolve_benchmark_config(config: ConfigDict | None = None) -> ConfigDict:
    """Merge benchmark defaults and validate the resolved configuration."""
    effective = deep_merge(DEFAULT_BENCHMARK_CONFIG, config or {})
    effective["quality_gates"] = {
        **effective.get("quality_gates", {}),
        "assessment_mode": effective["benchmark"]["assessment_mode"],
    }
    validate_benchmark_config(effective)
    return effective


def benchmark_matrix(config: ConfigDict) -> list[dict[str, Any]]:
    """Return the configured model x seed matrix."""
    benchmark = config["benchmark"]
    return [
        {"model": model, "seed": int(seed)}
        for seed in benchmark["seeds"]
        for model in benchmark["models"]
    ]


def run_benchmark(config: ConfigDict | None = None) -> dict[str, Any]:
    """Run the benchmark and write benchmark-level artifacts."""
    started = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    effective = resolve_benchmark_config(config)
    benchmark_id = build_benchmark_id(str(effective["benchmark"]["name"]), started)
    base_directory = Path(effective["outputs"]["base_directory"])
    benchmark_dir = base_directory / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(effective, benchmark_dir / "benchmark_config.yaml")
    LOGGER.info("benchmark_started", extra={"benchmark_id": benchmark_id})

    metadata = default_metadata()
    run_references: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    calibration_timings: dict[int, dict[str, float]] = {}
    continue_on_error = bool(effective["benchmark"]["continue_on_error"])

    unavailable = _preflight_models(effective["benchmark"]["models"])
    if unavailable and not continue_on_error:
        first = unavailable[0]
        raise ModelBackendUnavailable(first["message"])

    for seed in effective["benchmark"]["seeds"]:
        seed = int(seed)
        calibration_started = time.perf_counter()
        calibration_config = _calibration_config_for_seed(effective, seed)
        calibration = generate_calibration_dataset(config=calibration_config)
        calibration_seconds = time.perf_counter() - calibration_started
        split_started = time.perf_counter()
        train, holdout = split_train_holdout(
            calibration,
            holdout_fraction=float(effective["benchmark"]["holdout_fraction"]),
            seed=seed,
        )
        split_seconds = time.perf_counter() - split_started
        calibration_timings[seed] = {
            "calibration_seconds": float(calibration_seconds),
            "split_seconds": float(split_seconds),
        }
        save_calibration_splits(
            calibration,
            train,
            holdout,
            benchmark_dir / "calibration" / f"seed-{seed}",
            metadata=metadata,
        )

        for model in effective["benchmark"]["models"]:
            unavailable_for_model = next((item for item in unavailable if item["model"] == model), None)
            if unavailable_for_model is not None:
                failure = _record_failure(
                    benchmark_dir=benchmark_dir,
                    model=model,
                    seed=seed,
                    stage="preflight",
                    exc=ModelBackendUnavailable(unavailable_for_model["message"]),
                )
                failures.append(failure)
                run_references.append(_failed_run_reference(benchmark_id, model, seed, failure))
                if not continue_on_error:
                    _write_partial_outputs(
                        benchmark_dir,
                        effective,
                        benchmark_id,
                        started,
                        long_rows,
                        summary_rows,
                        run_references,
                        failures,
                    )
                    raise ModelBackendUnavailable(unavailable_for_model["message"])
                continue

            try:
                run_result = _run_one_model(
                    effective=effective,
                    benchmark_id=benchmark_id,
                    model=model,
                    seed=seed,
                    train=train,
                    holdout=holdout,
                    metadata=metadata,
                    calibration_timings=calibration_timings[seed],
                )
                run_references.append(run_result["run_reference"])
                summary_rows.append(run_result["summary_row"])
                long_rows.extend(run_result["long_rows"])
                _write_run_reference(benchmark_dir, model, seed, run_result["run_reference"])
            except Exception as exc:
                failure = _record_failure(benchmark_dir, model, seed, "run", exc)
                failures.append(failure)
                run_references.append(_failed_run_reference(benchmark_id, model, seed, failure))
                if not continue_on_error:
                    _write_partial_outputs(
                        benchmark_dir,
                        effective,
                        benchmark_id,
                        started,
                        long_rows,
                        summary_rows,
                        run_references,
                        failures,
                    )
                    raise

    completed_runs = len(summary_rows)
    failed_runs = len(failures)
    expected_runs = len(benchmark_matrix(effective))
    overall_status = "completed" if failed_runs == 0 else "completed_with_failures" if completed_runs else "failed"
    aggregate_by_model = aggregate_summary_by_model(summary_rows, failures, effective["benchmark"]["models"])
    ended = datetime.now(timezone.utc)
    outputs = _write_benchmark_outputs(
        benchmark_dir=benchmark_dir,
        config=effective,
        benchmark_id=benchmark_id,
        started_at=started,
        ended_at=ended,
        duration_seconds=float(time.perf_counter() - started_perf),
        run_references=run_references,
        long_rows=long_rows,
        summary_rows=summary_rows,
        failures=failures,
        aggregate_by_model=aggregate_by_model,
        overall_status=overall_status,
        expected_runs=expected_runs,
    )
    LOGGER.info(
        "benchmark_finished",
        extra={"benchmark_id": benchmark_id, "status": overall_status, "completed_runs": completed_runs, "failed_runs": failed_runs},
    )
    return {
        "benchmark_id": benchmark_id,
        "status": overall_status,
        "benchmark_dir": benchmark_dir,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "expected_runs": expected_runs,
        "runs": run_references,
        "summary_rows": summary_rows,
        "aggregate_by_model": aggregate_by_model,
        "failures": failures,
        "paths": outputs,
        "duration_seconds": float((ended - started).total_seconds()),
    }


def aggregate_summary_by_model(
    summary_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    models: list[str],
) -> dict[str, Any]:
    """Aggregate main benchmark metrics by model."""
    frame = pd.DataFrame(summary_rows)
    aggregates: dict[str, Any] = {}
    for model in models:
        model_frame = frame[frame["model"] == model] if not frame.empty else pd.DataFrame()
        status_counts = model_frame["status"].value_counts().to_dict() if "status" in model_frame else {}
        failed_count = sum(1 for failure in failures if failure["model"] == model)
        metrics: dict[str, Any] = {}
        for metric in MAIN_SUMMARY_METRICS:
            if metric not in model_frame:
                continue
            numeric = pd.to_numeric(model_frame[metric], errors="coerce").dropna()
            if numeric.empty:
                continue
            std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
            ci95 = None if len(numeric) < 2 else float(1.96 * std / math.sqrt(len(numeric)))
            metrics[metric] = {
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": std,
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "ci95_exploratory": ci95,
                "n": int(len(numeric)),
            }
        aggregates[model] = {
            "completed_runs": int(len(model_frame)),
            "approved": int(status_counts.get("approved", 0)),
            "quarantined": int(status_counts.get("quarantined", 0)),
            "rejected": int(status_counts.get("rejected", 0)),
            "failed": int(failed_count),
            "metrics": metrics,
            "ci_note": "95% confidence intervals are exploratory for small benchmark matrices.",
        }
    return aggregates


def _preflight_models(models: list[str]) -> list[dict[str, str]]:
    unavailable: list[dict[str, str]] = []
    for model in models:
        if model == "simple_gan":
            try:
                from synthetic_br_profiles_gan.models.gan import _require_tensorflow

                _require_tensorflow()
            except ModelBackendUnavailable as exc:
                unavailable.append({"model": model, "message": str(exc)})
        elif model == "ctgan":
            try:
                from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

                CTGANSynthesizer._ctgan_class()
            except ModelBackendUnavailable as exc:
                unavailable.append({"model": model, "message": str(exc)})
    return unavailable


def _calibration_config_for_seed(config: ConfigDict, seed: int) -> ConfigDict:
    calibration = dict(config.get("calibration", {}))
    calibration.update(
        {
            "seed": int(seed),
            "num_rows": int(config["benchmark"]["calibration_rows"]),
            "holdout_fraction": float(config["benchmark"]["holdout_fraction"]),
        }
    )
    return calibration


def _pipeline_config_for_run(config: ConfigDict, model: str, seed: int) -> ConfigDict:
    benchmark = config["benchmark"]
    artifacts_root = str(Path(config["outputs"]["base_directory"]).parent)
    generation = {
        "rows": int(benchmark["synthetic_rows"]),
        **config.get("generation", {}),
    }
    model_configs = {
        "programmatic": dict(config.get("models", {}).get("programmatic", {})),
        "simple_gan": dict(config.get("models", {}).get("simple_gan", {})),
        "ctgan": dict(config.get("models", {}).get("ctgan", {})),
    }
    model_configs.setdefault(model, {})
    model_seed = int(seed) + 100_003 if model == "programmatic" else int(seed)
    model_configs[model] = {**model_configs[model], "seed": model_seed}
    calibration = _calibration_config_for_seed(config, seed)
    return deep_merge(
        DEFAULT_PIPELINE_CONFIG,
        {
            "seed": int(seed),
            "artifacts_root": artifacts_root,
            "reference_date": str(benchmark["reference_date"]),
            "model": model,
            "calibration": calibration,
            "models": model_configs,
            "generation": generation,
            "evaluation": config.get("evaluation", {}),
            "quality_gates": {**config.get("quality_gates", {}), "assessment_mode": benchmark["assessment_mode"]},
            "export": {"xlsx": bool(config["outputs"].get("export_individual_xlsx", False)), "primary_format": "parquet"},
        },
    )


def _run_one_model(
    effective: ConfigDict,
    benchmark_id: str,
    model: str,
    seed: int,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    metadata: DatasetMetadata,
    calibration_timings: dict[str, float],
) -> dict[str, Any]:
    run_config = _pipeline_config_for_run(effective, model, seed)
    run_result = run_pipeline_on_splits(
        config=run_config,
        model_name=model,
        train=train,
        holdout=holdout,
        metadata=metadata,
    )
    run_reference = _run_reference(benchmark_id, model, seed, run_result)
    summary_row = summarize_run(benchmark_id, model, seed, run_result, calibration_timings)
    long_rows = flatten_run_metrics(benchmark_id, model, seed, run_result)
    return {"run_reference": run_reference, "summary_row": summary_row, "long_rows": long_rows}


def summarize_run(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
    calibration_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return one semilong summary row for a completed run."""
    evaluation = result["evaluation"]
    validation = result["validation"]
    manifest = result["manifest"]
    stage = result.get("stage_durations", {})
    privacy = evaluation.get("privacy", {})
    nearest = privacy.get("nearest_neighbor_train", {})
    dcr = nearest.get("distance_to_closest_record") or {}
    nndr = nearest.get("nearest_neighbor_distance_ratio") or {}
    renda = evaluation.get("against_holdout", {}).get("numeric", {}).get("Renda", {})
    genero = evaluation.get("against_holdout", {}).get("categorical", {}).get("Genero", {})
    correlation = evaluation.get("against_holdout", {}).get("correlations", {}).get("summary", {})
    calibration_timings = calibration_timings or {}
    return {
        "benchmark_id": benchmark_id,
        "run_id": result["run_id"],
        "model": model,
        "seed": int(seed),
        "status": result["status"],
        "duration_seconds": manifest.get("duration_seconds"),
        "calibration_seconds": calibration_timings.get("calibration_seconds"),
        "split_seconds": calibration_timings.get("split_seconds"),
        "training_seconds": stage.get("training_seconds"),
        "generation_seconds": stage.get("generation_seconds"),
        "postprocessing_seconds": result["generation"].get("postprocessing_seconds"),
        "validation_seconds": stage.get("validation_seconds"),
        "evaluation_seconds": stage.get("evaluation_seconds"),
        "export_seconds": stage.get("export_seconds"),
        "requested_rows": manifest.get("requested_rows"),
        "generated_rows": manifest.get("generated_rows"),
        "valid_rows": validation.get("valid_rows"),
        "invalid_rows": validation.get("invalid_rows"),
        "renda_wasserstein_normalized": renda.get("wasserstein_distance_normalized"),
        "renda_wasserstein": renda.get("wasserstein_distance"),
        "renda_ks": renda.get("ks_statistic"),
        "genero_tvd": genero.get("total_variation_distance"),
        "correlation_difference": correlation.get("mean_abs_difference"),
        "duplicate_row_rate": privacy.get("duplicate_row_rate"),
        "unique_combination_rate": privacy.get("unique_combination_rate"),
        "exact_train_match_rate": privacy.get("exact_train_match_rate"),
        "exact_holdout_match_rate": privacy.get("exact_holdout_match_rate"),
        "dcr": dcr.get("mean"),
        "nndr": nndr.get("mean"),
    }


def flatten_run_metrics(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten existing evaluation, validation, and gate outputs into long rows."""
    rows: list[dict[str, Any]] = []
    evaluation = result["evaluation"]
    validation = result["validation"]
    run_id = result["run_id"]
    status = result["status"]

    def add(
        metric_group: str,
        metric_name: str,
        column: str | None,
        value: Any,
        reference: Any = None,
        difference: Any = None,
        details: Any = None,
    ) -> None:
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "run_id": run_id,
                "model": model,
                "seed": int(seed),
                "status": status,
                "metric_group": metric_group,
                "metric_name": metric_name,
                "column": column,
                "value": value,
                "reference": reference,
                "difference": difference,
                "details": None if details is None else json.dumps(details, ensure_ascii=False, sort_keys=True, default=str),
            }
        )

    holdout_metrics = evaluation.get("against_holdout", {})
    for column in NUMERIC_COLUMNS:
        metric = holdout_metrics.get("numeric", {}).get(column, {})
        if not metric or "error" in metric:
            continue
        add("numeric", "mean", column, metric.get("synthetic", {}).get("mean"), metric.get("reference", {}).get("mean"), metric.get("absolute_mean_diff"))
        add("numeric", "median", column, metric.get("synthetic", {}).get("median"), metric.get("reference", {}).get("median"), metric.get("median_diff"))
        add("numeric", "std", column, metric.get("synthetic", {}).get("std"), metric.get("reference", {}).get("std"), metric.get("std_diff"))
        add("numeric", "relative_mean_diff", column, metric.get("relative_mean_diff"))
        add("numeric", "wasserstein_distance", column, metric.get("wasserstein_distance"))
        add("numeric", "wasserstein_distance_normalized", column, metric.get("wasserstein_distance_normalized"))
        add("numeric", "ks_statistic", column, metric.get("ks_statistic"))

    coverage = evaluation.get("privacy", {}).get("category_coverage_holdout", {})
    for column in CATEGORICAL_COLUMNS:
        metric = holdout_metrics.get("categorical", {}).get(column, {})
        if not metric:
            continue
        add("categorical", "total_variation_distance", column, metric.get("total_variation_distance"))
        add("categorical", "missing_categories_count", column, len(metric.get("missing_categories", [])), details=metric.get("missing_categories", []))
        add("categorical", "unexpected_categories_count", column, len(metric.get("unexpected_categories", [])), details=metric.get("unexpected_categories", []))
        add("categorical", "category_coverage_holdout", column, coverage.get(column))

    correlations = holdout_metrics.get("correlations", {})
    add("relationships", "correlation_mean_abs_difference", None, correlations.get("summary", {}).get("mean_abs_difference"))
    add("relationships", "correlation_max_abs_difference", None, correlations.get("summary", {}).get("max_abs_difference"))
    for method in ["pearson", "spearman"]:
        add("relationships", f"{method}_mean_abs_difference", None, _correlation_method_mean(correlations, method))

    for relationship, metric in holdout_metrics.get("categorical_relationships", {}).items():
        add("relationships", "categorical_relationship_tvd", relationship, metric.get("total_variation_distance"))
        add("relationships", "categorical_relationship_max_cell_difference", relationship, metric.get("max_cell_difference"))

    for group, metric in holdout_metrics.get("grouped_income", {}).items():
        add("relationships", "grouped_income_mean_abs_difference", group, _mean_dict_values(metric.get("absolute_difference", {})), details=metric.get("absolute_difference", {}))

    privacy = evaluation.get("privacy", {})
    add("privacy", "duplicate_row_rate", None, privacy.get("duplicate_row_rate"))
    add("privacy", "unique_combination_rate", None, privacy.get("unique_combination_rate"))
    add("privacy", "exact_train_match_rate", None, privacy.get("exact_train_match_rate"))
    add("privacy", "exact_holdout_match_rate", None, privacy.get("exact_holdout_match_rate"))
    train_nearest = privacy.get("nearest_neighbor_train", {})
    dcr = train_nearest.get("distance_to_closest_record") or {}
    nndr = train_nearest.get("nearest_neighbor_distance_ratio") or {}
    add("privacy", "dcr_train_mean", None, dcr.get("mean"))
    add("privacy", "dcr_train_min", None, dcr.get("min"))
    add("privacy", "nndr_train_mean", None, nndr.get("mean"))
    add("privacy", "columns_used", None, len(privacy.get("columns_used", [])), details=privacy.get("columns_used", []))
    add("privacy", "columns_excluded", None, len(privacy.get("excluded_columns", [])), details=privacy.get("excluded_columns", []))

    reason_counts = validation.get("reason_counts", {})
    add("validation", "invalid_rows", None, validation.get("invalid_rows"))
    add("validation", "valid_rows", None, validation.get("valid_rows"))
    add("validation", "null_required_fields", None, reason_counts.get("null_required_fields", 0))
    duplicated_identifier = sum(int(value) for key, value in reason_counts.items() if key.endswith("_duplicado") and key != "duplicated_rows")
    add("validation", "duplicated_identifiers", None, duplicated_identifier)
    add("validation", "rule_violation_types", None, len(reason_counts), details=reason_counts)
    for reason, count in reason_counts.items():
        add("validation", "rule_violation_count", reason, count)

    gates = result.get("quality_gates", {})
    add("quality_gates", "failure_count", None, len(gates.get("failures", [])), details=gates.get("failures", []))
    for metric, value in gates.get("metrics_checked", {}).items():
        add("quality_gates", metric, None, value)

    return rows


def _correlation_method_mean(correlations: dict[str, Any], method: str) -> float | None:
    diff = correlations.get(method, {}).get("absolute_difference", {})
    values: list[float] = []
    for row in diff.values():
        if isinstance(row, dict):
            values.extend(float(value) for value in row.values() if pd.notna(value))
    return float(sum(values) / len(values)) if values else None


def _mean_dict_values(values: dict[str, Any]) -> float | None:
    numeric = [float(value) for value in values.values() if value is not None and pd.notna(value)]
    return float(sum(numeric) / len(numeric)) if numeric else None


def _run_reference(benchmark_id: str, model: str, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    paths = result["paths"]
    return {
        "benchmark_id": benchmark_id,
        "run_id": result["run_id"],
        "model": model,
        "seed": int(seed),
        "status": result["status"],
        "manifest": str(paths["manifest"]),
        "root_manifest": str(paths["root_manifest"]),
        "dataset_parquet": str(paths["dataset_parquet"]),
        "evaluation": str(paths["evaluation"]),
        "quality_gates": str(paths["quality_gates"]),
    }


def _failed_run_reference(benchmark_id: str, model: str, seed: int, failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "run_id": None,
        "model": model,
        "seed": int(seed),
        "status": "failed",
        "failure": failure,
    }


def _write_run_reference(benchmark_dir: Path, model: str, seed: int, reference: dict[str, Any]) -> Path:
    return write_json(reference, benchmark_dir / "runs" / model / f"seed-{seed}" / "run-reference.json")


def _record_failure(benchmark_dir: Path, model: str, seed: int, stage: str, exc: BaseException) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    diagnostics_dir = benchmark_dir / "diagnostics" / model / f"seed-{seed}"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    traceback_path = diagnostics_dir / f"{stage}.traceback.txt"
    with traceback_path.open("w", encoding="utf-8") as file:
        file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {
        "model": model,
        "seed": int(seed),
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "timestamp_utc": timestamp,
        "traceback_path": str(traceback_path),
    }


def _write_partial_outputs(
    benchmark_dir: Path,
    config: ConfigDict,
    benchmark_id: str,
    started_at: datetime,
    long_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    run_references: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    aggregate = aggregate_summary_by_model(summary_rows, failures, config["benchmark"]["models"])
    _write_benchmark_outputs(
        benchmark_dir=benchmark_dir,
        config=config,
        benchmark_id=benchmark_id,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        duration_seconds=0.0,
        run_references=run_references,
        long_rows=long_rows,
        summary_rows=summary_rows,
        failures=failures,
        aggregate_by_model=aggregate,
        overall_status="completed_with_failures" if summary_rows else "failed",
        expected_runs=len(benchmark_matrix(config)),
    )


def _write_benchmark_outputs(
    benchmark_dir: Path,
    config: ConfigDict,
    benchmark_id: str,
    started_at: datetime,
    ended_at: datetime,
    duration_seconds: float,
    run_references: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    aggregate_by_model: dict[str, Any],
    overall_status: str,
    expected_runs: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    export_csv = bool(config["outputs"].get("export_csv", True))
    export_parquet = bool(config["outputs"].get("export_parquet", True))
    export_json = bool(config["outputs"].get("export_json", True))
    long_frame = pd.DataFrame(long_rows, columns=LONG_RESULT_COLUMNS)
    summary_frame = pd.DataFrame(summary_rows, columns=RUN_SUMMARY_COLUMNS)
    if export_parquet:
        outputs["results_parquet"] = benchmark_dir / "results.parquet"
        outputs["run_summary_parquet"] = benchmark_dir / "run_summary.parquet"
        long_frame.to_parquet(outputs["results_parquet"], index=False)
        summary_frame.to_parquet(outputs["run_summary_parquet"], index=False)
    if export_csv:
        outputs["results_csv"] = benchmark_dir / "results.csv"
        outputs["run_summary_csv"] = benchmark_dir / "run_summary.csv"
        long_frame.to_csv(outputs["results_csv"], index=False)
        summary_frame.to_csv(outputs["run_summary_csv"], index=False)
    summary_payload = {
        "benchmark_id": benchmark_id,
        "status": overall_status,
        "expected_runs": int(expected_runs),
        "completed_runs": int(len(summary_rows)),
        "failed_runs": int(len(failures)),
        "aggregate_by_model": aggregate_by_model,
        "interpretation": "Exploratory pilot benchmark; do not treat a single metric as a final model ranking.",
    }
    if export_json:
        outputs["summary_json"] = write_json(summary_payload, benchmark_dir / "summary.json")
        outputs["runs_json"] = write_json({"runs": run_references}, benchmark_dir / "runs.json")
        outputs["failures_json"] = write_json({"failures": failures}, benchmark_dir / "failures.json")

    manifest = _benchmark_manifest(
        benchmark_id=benchmark_id,
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        expected_runs=expected_runs,
        completed_runs=len(summary_rows),
        failed_runs=len(failures),
        status=overall_status,
        artifact_paths={**outputs, "benchmark_config": benchmark_dir / "benchmark_config.yaml"},
    )
    outputs["benchmark_manifest"] = write_json(manifest, benchmark_dir / "benchmark_manifest.json")
    return outputs


def _benchmark_manifest(
    benchmark_id: str,
    config: ConfigDict,
    started_at: datetime,
    ended_at: datetime,
    duration_seconds: float,
    expected_runs: int,
    completed_runs: int,
    failed_runs: int,
    status: str,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    artifact_hashes = {
        key: hash_file(path)
        for key, path in artifact_paths.items()
        if path.exists() and path.is_file()
    }
    return {
        "benchmark_id": benchmark_id,
        "timestamp_utc": started_at.astimezone(timezone.utc).isoformat(),
        "ended_at_utc": ended_at.astimezone(timezone.utc).isoformat(),
        "duration_seconds": float(duration_seconds),
        "config_hash": config_hash(config),
        "models": config["benchmark"]["models"],
        "seeds": config["benchmark"]["seeds"],
        "expected_runs": int(expected_runs),
        "completed_runs": int(completed_runs),
        "failed_runs": int(failed_runs),
        "status": status,
        "environment": environment_info(),
        "git_commit": get_git_commit(Path.cwd()),
        "artifact_hashes": artifact_hashes,
        "methodology": {
            "same_calibration_per_seed": True,
            "same_train_holdout_per_seed": True,
            "assessment_mode": config["benchmark"]["assessment_mode"],
            "parallelism": config.get("execution", {}).get("parallelism", 1),
        },
    }
