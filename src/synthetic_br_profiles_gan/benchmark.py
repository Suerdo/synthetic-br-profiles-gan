"""Benchmark orchestration for comparing tabular synthesizers."""

from __future__ import annotations

import logging
import json
import math
import os
import re
import threading
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


class ResourceLimitExceeded(PipelineError):
    """Raised when a configured benchmark resource limit is exceeded."""

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
    "execution": {"parallelism": 1, "warmup_backends": False, "rotate_model_order_by_seed": False},
    "resource_limits": {
        "max_training_seconds_per_run": None,
        "max_total_seconds_per_run": None,
        "max_peak_memory_mb": None,
        "stop_larger_sizes_after_resource_failure": True,
    },
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
    "peak_memory_mb",
    "model_size_mb",
]
LONG_RESULT_COLUMNS = [
    "benchmark_id",
    "run_id",
    "model",
    "seed",
    "train_size",
    "holdout_size",
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
    "train_size",
    "holdout_size",
    "calibration_rows",
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
    "memory_before_training_mb",
    "memory_after_training_mb",
    "peak_memory_mb",
    "model_size_mb",
    "artifact_size_mb",
    "cpu_count",
    "thread_count",
    "backend_warmup_seconds",
    "batches_per_epoch",
    "generator_updates",
    "discriminator_updates",
    "mean_epoch_seconds",
    "ctgan_batches_per_epoch_inferred",
    "ctgan_total_batches_inferred",
    "resource_limited",
]


def build_benchmark_id(name: str, timestamp: datetime | None = None) -> str:
    """Build a stable benchmark id prefix plus a timestamp/short suffix."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip().lower()).strip("-") or "benchmark"
    return f"{slug}-{build_run_id(timestamp)}"


def resolve_benchmark_config(config: ConfigDict | None = None) -> ConfigDict:
    """Merge benchmark defaults and validate the resolved configuration."""
    effective = deep_merge(DEFAULT_BENCHMARK_CONFIG, config or {})
    if config and isinstance(config.get("benchmark"), dict) and "train_sizes" in config["benchmark"] and "calibration_rows" not in config["benchmark"]:
        effective["benchmark"].pop("calibration_rows", None)
    effective["quality_gates"] = {
        **effective.get("quality_gates", {}),
        "assessment_mode": effective["benchmark"]["assessment_mode"],
    }
    validate_benchmark_config(effective)
    return effective


def benchmark_matrix(config: ConfigDict) -> list[dict[str, Any]]:
    """Return the configured model x seed x optional train-size matrix."""
    benchmark = config["benchmark"]
    rows: list[dict[str, Any]] = []
    if "train_sizes" in benchmark:
        for seed in benchmark["seeds"]:
            for train_size in benchmark["train_sizes"]:
                holdout_size = holdout_rows_for_train_size(int(train_size), float(benchmark["holdout_fraction"]))
                for model in benchmark["models"]:
                    rows.append(
                        {
                            "model": model,
                            "seed": int(seed),
                            "train_size": int(train_size),
                            "holdout_size": int(holdout_size),
                            "calibration_rows": int(train_size) + int(holdout_size),
                        }
                    )
    else:
        for seed in benchmark["seeds"]:
            for model in benchmark["models"]:
                rows.append({"model": model, "seed": int(seed), "train_size": None, "holdout_size": None})
    return rows


def holdout_rows_for_train_size(train_size: int, holdout_fraction: float) -> int:
    """Return the exact holdout size implied by a target training size."""
    value = int(train_size) * float(holdout_fraction) / (1.0 - float(holdout_fraction))
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_size and holdout_fraction do not produce an exact integer holdout size.")
    return rounded


def train_size_specs(config: ConfigDict) -> list[dict[str, int | None]]:
    """Return the configured train-size specifications."""
    benchmark = config["benchmark"]
    if "train_sizes" not in benchmark:
        return [{"train_size": None, "holdout_size": None, "calibration_rows": int(benchmark["calibration_rows"])}]
    return [
        {
            "train_size": int(train_size),
            "holdout_size": holdout_rows_for_train_size(int(train_size), float(benchmark["holdout_fraction"])),
            "calibration_rows": int(train_size) + holdout_rows_for_train_size(int(train_size), float(benchmark["holdout_fraction"])),
        }
        for train_size in benchmark["train_sizes"]
    ]


def rotate_models_for_seed(models: list[str], seed_index: int, rotate: bool) -> list[str]:
    """Rotate model execution order by seed to reduce cache/warm-up bias."""
    if not rotate or not models:
        return list(models)
    offset = seed_index % len(models)
    return [*models[offset:], *models[:offset]]


class ResourceMonitor:
    """Sample resident process memory during one benchmark run."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = float(interval_seconds)
        self.available = False
        self.peak_memory_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any | None = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self.available = True
        except ImportError:
            self._process = None

    def __enter__(self) -> "ResourceMonitor":
        first = self.current_memory_mb()
        self.peak_memory_mb = first
        if self.available:
            self._thread = threading.Thread(target=self._sample_loop, name="benchmark-resource-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback_obj) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        last = self.current_memory_mb()
        if last is not None:
            self.peak_memory_mb = max(last, self.peak_memory_mb or last)

    def current_memory_mb(self) -> float | None:
        """Return current resident memory in MiB when psutil is available."""
        if self._process is None:
            return None
        return float(self._process.memory_info().rss / (1024 * 1024))

    def thread_count(self) -> int | None:
        """Return current process thread count when psutil is available."""
        if self._process is None:
            return None
        return int(self._process.num_threads())

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = self.current_memory_mb()
            if current is not None:
                self.peak_memory_mb = max(current, self.peak_memory_mb or current)


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
    calibration_timings: dict[tuple[int, int | None], dict[str, float]] = {}
    continue_on_error = bool(effective["benchmark"]["continue_on_error"])
    size_specs = train_size_specs(effective)
    rotate_order = bool(effective.get("execution", {}).get("rotate_model_order_by_seed", False))
    resource_stop_sizes: dict[str, int] = {}

    unavailable = _preflight_models(effective["benchmark"]["models"])
    if unavailable and not continue_on_error:
        first = unavailable[0]
        raise ModelBackendUnavailable(first["message"])
    warmup_seconds = _warmup_backends(effective["benchmark"]["models"]) if bool(effective.get("execution", {}).get("warmup_backends", False)) else {}

    for seed_index, seed in enumerate(effective["benchmark"]["seeds"]):
        seed = int(seed)
        model_order = rotate_models_for_seed(effective["benchmark"]["models"], seed_index, rotate_order)
        for size_spec in size_specs:
            train_size = size_spec["train_size"]
            holdout_size = size_spec["holdout_size"]
            calibration_rows = int(size_spec["calibration_rows"])
            calibration_started = time.perf_counter()
            calibration_config = _calibration_config_for_seed(effective, seed, calibration_rows=calibration_rows)
            calibration = generate_calibration_dataset(config=calibration_config)
            calibration_seconds = time.perf_counter() - calibration_started
            split_started = time.perf_counter()
            if train_size is not None and holdout_size is not None:
                train, holdout = split_train_holdout(
                    calibration,
                    holdout_fraction=float(effective["benchmark"]["holdout_fraction"]),
                    seed=seed,
                    train_rows=int(train_size),
                    holdout_rows=int(holdout_size),
                )
            else:
                train, holdout = split_train_holdout(
                    calibration,
                    holdout_fraction=float(effective["benchmark"]["holdout_fraction"]),
                    seed=seed,
                )
                train_size = int(train.shape[0])
                holdout_size = int(holdout.shape[0])
            split_seconds = time.perf_counter() - split_started
            calibration_key = (seed, None if "train_sizes" not in effective["benchmark"] else int(train_size))
            calibration_timings[calibration_key] = {
                "calibration_seconds": float(calibration_seconds),
                "split_seconds": float(split_seconds),
            }
            save_calibration_splits(
                calibration,
                train,
                holdout,
                _calibration_output_dir(benchmark_dir, seed, calibration_key[1]),
                metadata=metadata,
            )

            for model in model_order:
                if calibration_key[1] is not None and model in resource_stop_sizes and int(calibration_key[1]) > resource_stop_sizes[model]:
                    exc = ResourceLimitExceeded(
                        f"Skipped train_size={calibration_key[1]} after resource limit at train_size={resource_stop_sizes[model]}."
                    )
                    failure = _record_failure(benchmark_dir, model, seed, "resource_limit_skip", exc, train_size=calibration_key[1], holdout_size=int(holdout_size))
                    failure["status"] = "resource_limited"
                    failures.append(failure)
                    run_references.append(_failed_run_reference(benchmark_id, model, seed, failure, train_size=calibration_key[1], holdout_size=int(holdout_size)))
                    continue

                unavailable_for_model = next((item for item in unavailable if item["model"] == model), None)
                if unavailable_for_model is not None:
                    failure = _record_failure(
                        benchmark_dir=benchmark_dir,
                        model=model,
                        seed=seed,
                        stage="preflight",
                        exc=ModelBackendUnavailable(unavailable_for_model["message"]),
                        train_size=calibration_key[1],
                        holdout_size=int(holdout_size),
                    )
                    failures.append(failure)
                    run_references.append(
                        _failed_run_reference(benchmark_id, model, seed, failure, train_size=calibration_key[1], holdout_size=int(holdout_size))
                    )
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
                        calibration_timings=calibration_timings[calibration_key],
                        train_size=calibration_key[1],
                        holdout_size=int(holdout_size),
                        calibration_rows=calibration_rows,
                        backend_warmup_seconds=warmup_seconds.get(model),
                    )
                    limit_failure = _resource_limit_failure(
                        benchmark_dir=benchmark_dir,
                        model=model,
                        seed=seed,
                        train_size=calibration_key[1],
                        holdout_size=int(holdout_size),
                        summary_row=run_result["summary_row"],
                        limits=effective.get("resource_limits", {}),
                    )
                    if limit_failure is not None:
                        run_result["summary_row"]["resource_limited"] = True
                        run_result["run_reference"]["resource_limited"] = True
                        failures.append(limit_failure)
                        if calibration_key[1] is not None and bool(
                            effective.get("resource_limits", {}).get("stop_larger_sizes_after_resource_failure", True)
                        ):
                            resource_stop_sizes[model] = min(resource_stop_sizes.get(model, int(calibration_key[1])), int(calibration_key[1]))
                    run_references.append(run_result["run_reference"])
                    summary_rows.append(run_result["summary_row"])
                    long_rows.extend(run_result["long_rows"])
                    _write_run_reference(benchmark_dir, model, seed, run_result["run_reference"], train_size=calibration_key[1])
                    if limit_failure is not None and not continue_on_error:
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
                        raise ResourceLimitExceeded(limit_failure["message"])
                except ResourceLimitExceeded:
                    raise
                except Exception as exc:
                    failure = _record_failure(benchmark_dir, model, seed, "run", exc, train_size=calibration_key[1], holdout_size=int(holdout_size))
                    failures.append(failure)
                    run_references.append(
                        _failed_run_reference(benchmark_id, model, seed, failure, train_size=calibration_key[1], holdout_size=int(holdout_size))
                    )
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
    summary_rows = _sort_summary_rows(summary_rows)
    long_rows = _sort_long_rows(long_rows)
    run_references = _sort_run_references(run_references)
    aggregate_by_model = aggregate_summary_by_model(summary_rows, failures, effective["benchmark"]["models"])
    aggregate_by_model_and_size = aggregate_summary_by_model_and_size(summary_rows, failures, effective["benchmark"]["models"])
    marginal_gains = calculate_marginal_gains(summary_rows)
    scalability_limits = calculate_scalability_limits(summary_rows, failures, effective["benchmark"]["models"], effective["benchmark"].get("train_sizes", []))
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
        aggregate_by_model_and_size=aggregate_by_model_and_size,
        marginal_gains=marginal_gains,
        scalability_limits=scalability_limits,
        warmup_seconds=warmup_seconds,
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
        "aggregate_by_model_and_size": aggregate_by_model_and_size,
        "marginal_gains": marginal_gains,
        "scalability_limits": scalability_limits,
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


def aggregate_summary_by_model_and_size(
    summary_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    models: list[str],
) -> dict[str, Any]:
    """Aggregate main benchmark metrics by model and train size."""
    frame = pd.DataFrame(summary_rows)
    train_sizes = sorted(
        int(size)
        for size in frame.get("train_size", pd.Series(dtype="float")).dropna().unique().tolist()
    ) if not frame.empty and "train_size" in frame else []
    aggregates: dict[str, Any] = {}
    for model in models:
        model_payload: dict[str, Any] = {}
        for train_size in train_sizes:
            size_frame = frame[(frame["model"] == model) & (frame["train_size"] == train_size)] if not frame.empty else pd.DataFrame()
            status_counts = size_frame["status"].value_counts().to_dict() if "status" in size_frame else {}
            failed_count = sum(1 for failure in failures if failure["model"] == model and failure.get("train_size") == train_size)
            metrics: dict[str, Any] = {}
            for metric in [*MAIN_SUMMARY_METRICS, "peak_memory_mb", "model_size_mb", "artifact_size_mb", "batches_per_epoch", "generator_updates", "discriminator_updates"]:
                if metric not in size_frame:
                    continue
                numeric = pd.to_numeric(size_frame[metric], errors="coerce").dropna()
                if numeric.empty:
                    continue
                std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
                metrics[metric] = {
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "std": std,
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "ci95_exploratory": None if len(numeric) < 2 else float(1.96 * std / math.sqrt(len(numeric))),
                    "n": int(len(numeric)),
                }
            model_payload[str(train_size)] = {
                "completed_runs": int(len(size_frame)),
                "approved": int(status_counts.get("approved", 0)),
                "quarantined": int(status_counts.get("quarantined", 0)),
                "rejected": int(status_counts.get("rejected", 0)),
                "failed": int(failed_count),
                "metrics": metrics,
            }
        aggregates[model] = model_payload
    return aggregates


def calculate_marginal_gains(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate exploratory marginal changes between configured train sizes."""
    frame = pd.DataFrame(summary_rows)
    if frame.empty or "train_size" not in frame:
        return []
    metrics = [
        "renda_wasserstein_normalized",
        "renda_ks",
        "genero_tvd",
        "correlation_difference",
        "training_seconds",
        "peak_memory_mb",
        "model_size_mb",
        "artifact_size_mb",
    ]
    rows: list[dict[str, Any]] = []
    for model, model_frame in frame.groupby("model"):
        by_size: dict[int, dict[str, float]] = {}
        for train_size, size_frame in model_frame.dropna(subset=["train_size"]).groupby("train_size"):
            by_size[int(train_size)] = {}
            for metric in metrics:
                if metric not in size_frame:
                    continue
                numeric = pd.to_numeric(size_frame.get(metric), errors="coerce").dropna()
                if not numeric.empty:
                    by_size[int(train_size)][metric] = float(numeric.mean())
        for smaller, larger in [(1000, 5000), (5000, 20000), (1000, 20000)]:
            if smaller not in by_size or larger not in by_size:
                continue
            row: dict[str, Any] = {"model": model, "comparison": f"{smaller}_to_{larger}", "from_train_size": smaller, "to_train_size": larger}
            for metric in metrics:
                before = by_size[smaller].get(metric)
                after = by_size[larger].get(metric)
                if before is None or after is None:
                    continue
                change = float(after - before)
                row[f"{metric}_change"] = change
                if abs(before) > 1e-12:
                    row[f"{metric}_percent_change"] = float(change / abs(before))
            quality_gain = _quality_distance_gain(row)
            time_change = row.get("training_seconds_change")
            memory_change = row.get("peak_memory_mb_change")
            model_size_change = row.get("model_size_mb_change")
            row["quality_gain_by_training_second"] = _safe_ratio(quality_gain, time_change)
            row["quality_gain_by_peak_memory_mb"] = _safe_ratio(quality_gain, memory_change)
            row["quality_gain_by_model_mb"] = _safe_ratio(quality_gain, model_size_change)
            rows.append(row)
    return rows


def calculate_scalability_limits(
    summary_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    models: list[str],
    train_sizes: list[int],
) -> list[dict[str, Any]]:
    """Summarize observed successful train sizes per model for this environment."""
    if not train_sizes:
        return []
    frame = pd.DataFrame(summary_rows)
    sorted_sizes = sorted(int(size) for size in train_sizes)
    payload: list[dict[str, Any]] = []
    for model in models:
        successful = []
        resource_limited = False
        if not frame.empty:
            model_frame = frame[(frame["model"] == model) & (frame["resource_limited"] != True)]  # noqa: E712
            successful = sorted(int(size) for size in model_frame.get("train_size", pd.Series(dtype="float")).dropna().unique().tolist())
        failed_sizes = sorted(
            int(failure["train_size"])
            for failure in failures
            if failure["model"] == model and failure.get("train_size") is not None
        )
        resource_limited = any(failure.get("status") == "resource_limited" for failure in failures if failure["model"] == model)
        first_failed_size = next((size for size in sorted_sizes if size in failed_sizes and size not in successful), None)
        largest = max(successful) if successful else None
        if largest is None:
            conclusion = "Nenhum tamanho de treinamento foi concluído com sucesso neste ambiente."
        elif first_failed_size is None:
            conclusion = f"Executado com sucesso com até {largest:,} registros neste ambiente.".replace(",", ".")
        elif largest > first_failed_size:
            conclusion = (
                f"Houve falha em {first_failed_size:,} registros, mas também houve sucesso posterior até {largest:,}; "
                "o comportamento observado indica instabilidade neste ambiente."
            ).replace(",", ".")
        else:
            conclusion = (
                f"O limite observado está entre {largest:,} e {first_failed_size:,} registros neste ambiente; "
                "o valor exato não foi determinado."
            ).replace(",", ".")
        payload.append(
            {
                "model": model,
                "tested_train_sizes": sorted_sizes,
                "successful_train_sizes": successful,
                "largest_tested_successful_size": largest,
                "first_failed_size": first_failed_size,
                "resource_limited": resource_limited,
                "conclusion": conclusion,
            }
        )
    return payload


def _quality_distance_gain(row: dict[str, Any]) -> float | None:
    gains = [
        -float(row[key])
        for key in [
            "renda_wasserstein_normalized_change",
            "renda_ks_change",
            "genero_tvd_change",
            "correlation_difference_change",
        ]
        if row.get(key) is not None and pd.notna(row.get(key))
    ]
    return float(sum(gains) / len(gains)) if gains else None


def _safe_ratio(numerator: float | None, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    try:
        denominator_float = float(denominator)
    except (TypeError, ValueError):
        return None
    if abs(denominator_float) <= 1e-12:
        return None
    return float(numerator / denominator_float)


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


def _warmup_backends(models: list[str]) -> dict[str, float]:
    """Load optional model backends once and record warm-up time separately."""
    timings: dict[str, float] = {}
    for model in models:
        started = time.perf_counter()
        try:
            if model == "simple_gan":
                from synthetic_br_profiles_gan.models.gan import _require_tensorflow

                _require_tensorflow()
            elif model == "ctgan":
                from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

                CTGANSynthesizer._ctgan_class()
            else:
                timings[model] = 0.0
                continue
            timings[model] = float(time.perf_counter() - started)
        except ModelBackendUnavailable:
            timings[model] = float(time.perf_counter() - started)
    return timings


def _calibration_output_dir(benchmark_dir: Path, seed: int, train_size: int | None) -> Path:
    output_dir = benchmark_dir / "calibration" / f"seed-{seed}"
    if train_size is not None:
        output_dir = output_dir / f"train-{train_size}"
    return output_dir


def _sort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row.get("train_size") is None, row.get("train_size") or 0, str(row.get("model")), row.get("seed") or 0))


def _sort_long_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("train_size") is None,
            row.get("train_size") or 0,
            str(row.get("model")),
            row.get("seed") or 0,
            str(row.get("metric_group")),
            str(row.get("metric_name")),
            str(row.get("column")),
        ),
    )


def _sort_run_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row.get("train_size") is None, row.get("train_size") or 0, str(row.get("model")), row.get("seed") or 0))


def _calibration_config_for_seed(config: ConfigDict, seed: int, calibration_rows: int | None = None) -> ConfigDict:
    calibration = dict(config.get("calibration", {}))
    calibration.update(
        {
            "seed": int(seed),
            "num_rows": int(calibration_rows if calibration_rows is not None else config["benchmark"]["calibration_rows"]),
            "holdout_fraction": float(config["benchmark"]["holdout_fraction"]),
        }
    )
    return calibration


def _pipeline_config_for_run(config: ConfigDict, model: str, seed: int, calibration_rows: int | None = None) -> ConfigDict:
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
    calibration = _calibration_config_for_seed(config, seed, calibration_rows=calibration_rows)
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
    train_size: int | None,
    holdout_size: int,
    calibration_rows: int,
    backend_warmup_seconds: float | None = None,
) -> dict[str, Any]:
    run_config = _pipeline_config_for_run(effective, model, seed, calibration_rows=calibration_rows)
    with ResourceMonitor() as monitor:
        run_result = run_pipeline_on_splits(
            config=run_config,
            model_name=model,
            train=train,
            holdout=holdout,
            metadata=metadata,
            resource_probe=monitor.current_memory_mb,
        )
    run_result["resource_monitor"] = {
        "peak_memory_mb": monitor.peak_memory_mb,
        "cpu_count": os.cpu_count(),
        "thread_count": monitor.thread_count(),
        "psutil_available": monitor.available,
    }
    run_result["resolved_run_config"] = run_config
    run_reference = _run_reference(benchmark_id, model, seed, run_result, train_size=train_size, holdout_size=holdout_size)
    summary_row = summarize_run(
        benchmark_id,
        model,
        seed,
        run_result,
        calibration_timings,
        train_size=train_size,
        holdout_size=holdout_size,
        calibration_rows=calibration_rows,
        backend_warmup_seconds=backend_warmup_seconds,
    )
    long_rows = flatten_run_metrics(benchmark_id, model, seed, run_result, train_size=train_size, holdout_size=holdout_size)
    return {"run_reference": run_reference, "summary_row": summary_row, "long_rows": long_rows}


def summarize_run(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
    calibration_timings: dict[str, float] | None = None,
    train_size: int | None = None,
    holdout_size: int | None = None,
    calibration_rows: int | None = None,
    backend_warmup_seconds: float | None = None,
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
    resources = result.get("stage_resources", {})
    monitor = result.get("resource_monitor", {})
    simple_history = _read_simple_gan_history(result.get("model_dir"))
    ctgan_training = _ctgan_training_inference(model, result, train_size)
    return {
        "benchmark_id": benchmark_id,
        "run_id": result["run_id"],
        "model": model,
        "seed": int(seed),
        "train_size": train_size,
        "holdout_size": holdout_size,
        "calibration_rows": calibration_rows,
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
        "memory_before_training_mb": resources.get("memory_before_training_mb"),
        "memory_after_training_mb": resources.get("memory_after_training_mb"),
        "peak_memory_mb": monitor.get("peak_memory_mb"),
        "model_size_mb": _path_size_mb(result.get("model_dir")),
        "artifact_size_mb": _run_artifact_size_mb(result),
        "cpu_count": monitor.get("cpu_count"),
        "thread_count": monitor.get("thread_count"),
        "backend_warmup_seconds": backend_warmup_seconds,
        "batches_per_epoch": simple_history.get("batches_per_epoch"),
        "generator_updates": simple_history.get("total_generator_updates"),
        "discriminator_updates": simple_history.get("total_discriminator_updates"),
        "mean_epoch_seconds": _mean_epoch_seconds(simple_history),
        "ctgan_batches_per_epoch_inferred": ctgan_training.get("batches_per_epoch_inferred"),
        "ctgan_total_batches_inferred": ctgan_training.get("total_batches_inferred"),
        "resource_limited": False,
    }


def flatten_run_metrics(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
    train_size: int | None = None,
    holdout_size: int | None = None,
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
                "train_size": train_size,
                "holdout_size": holdout_size,
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

    resources = result.get("stage_resources", {})
    monitor = result.get("resource_monitor", {})
    add("resources", "memory_before_training_mb", None, resources.get("memory_before_training_mb"))
    add("resources", "memory_after_training_mb", None, resources.get("memory_after_training_mb"))
    add("resources", "peak_memory_mb", None, monitor.get("peak_memory_mb"))
    add("resources", "model_size_mb", None, _path_size_mb(result.get("model_dir")))
    add("resources", "artifact_size_mb", None, _run_artifact_size_mb(result))

    simple_history = _read_simple_gan_history(result.get("model_dir"))
    if simple_history:
        add("training", "batches_per_epoch", None, simple_history.get("batches_per_epoch"))
        add("training", "generator_updates", None, simple_history.get("total_generator_updates"))
        add("training", "discriminator_updates", None, simple_history.get("total_discriminator_updates"))
        add("training", "mean_epoch_seconds", None, _mean_epoch_seconds(simple_history))
    ctgan_training = _ctgan_training_inference(model, result, train_size)
    if ctgan_training:
        add("training", "ctgan_batches_per_epoch_inferred", None, ctgan_training.get("batches_per_epoch_inferred"))
        add("training", "ctgan_total_batches_inferred", None, ctgan_training.get("total_batches_inferred"))

    return rows


def _correlation_method_mean(correlations: dict[str, Any], method: str) -> float | None:
    diff = correlations.get(method, {}).get("absolute_difference", {})
    values: list[float] = []
    for row in diff.values():
        if isinstance(row, dict):
            values.extend(float(value) for value in row.values() if pd.notna(value))
    return float(sum(values) / len(values)) if values else None


def _read_simple_gan_history(model_dir: Any) -> dict[str, Any]:
    path = Path(model_dir) / "training_history.json" if model_dir is not None else None
    if path is None or not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("simple_gan_history_unavailable", extra={"path": str(path)})
        return {}


def _mean_epoch_seconds(history: dict[str, Any]) -> float | None:
    epochs = history.get("epochs_history", [])
    values = [
        float(item["duration_seconds"])
        for item in epochs
        if isinstance(item, dict) and item.get("duration_seconds") is not None and pd.notna(item.get("duration_seconds"))
    ]
    return float(sum(values) / len(values)) if values else None


def _ctgan_training_inference(model: str, result: dict[str, Any], train_size: int | None) -> dict[str, int]:
    if model != "ctgan" or train_size is None:
        return {}
    config = result.get("resolved_run_config", {}).get("models", {}).get("ctgan", {})
    try:
        batch_size = int(config["batch_size"])
        epochs = int(config["epochs"])
    except (KeyError, TypeError, ValueError):
        return {}
    batches = int(math.ceil(int(train_size) / batch_size))
    return {"batches_per_epoch_inferred": batches, "total_batches_inferred": int(batches * epochs)}


def _path_size_mb(path: Any) -> float | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return float(root.stat().st_size / (1024 * 1024))
    total = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    return float(total / (1024 * 1024))


def _run_artifact_size_mb(result: dict[str, Any]) -> float | None:
    root_manifest = result.get("paths", {}).get("root_manifest")
    if root_manifest is None:
        return None
    return _path_size_mb(Path(root_manifest).parent)


def _mean_dict_values(values: dict[str, Any]) -> float | None:
    numeric = [float(value) for value in values.values() if value is not None and pd.notna(value)]
    return float(sum(numeric) / len(numeric)) if numeric else None


def _run_reference(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
    train_size: int | None = None,
    holdout_size: int | None = None,
) -> dict[str, Any]:
    paths = result["paths"]
    return {
        "benchmark_id": benchmark_id,
        "run_id": result["run_id"],
        "model": model,
        "seed": int(seed),
        "train_size": train_size,
        "holdout_size": holdout_size,
        "status": result["status"],
        "manifest": str(paths["manifest"]),
        "root_manifest": str(paths["root_manifest"]),
        "dataset_parquet": str(paths["dataset_parquet"]),
        "evaluation": str(paths["evaluation"]),
        "quality_gates": str(paths["quality_gates"]),
    }


def _failed_run_reference(
    benchmark_id: str,
    model: str,
    seed: int,
    failure: dict[str, Any],
    train_size: int | None = None,
    holdout_size: int | None = None,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "run_id": None,
        "model": model,
        "seed": int(seed),
        "train_size": train_size,
        "holdout_size": holdout_size,
        "status": "failed",
        "failure": failure,
    }


def _write_run_reference(benchmark_dir: Path, model: str, seed: int, reference: dict[str, Any], train_size: int | None = None) -> Path:
    output_dir = benchmark_dir / "runs" / model / f"seed-{seed}"
    if train_size is not None:
        output_dir = output_dir / f"train-{train_size}"
    return write_json(reference, output_dir / "run-reference.json")


def _record_failure(
    benchmark_dir: Path,
    model: str,
    seed: int,
    stage: str,
    exc: BaseException,
    train_size: int | None = None,
    holdout_size: int | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    diagnostics_dir = benchmark_dir / "diagnostics" / model / f"seed-{seed}"
    if train_size is not None:
        diagnostics_dir = diagnostics_dir / f"train-{train_size}"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    traceback_path = diagnostics_dir / f"{stage}.traceback.txt"
    with traceback_path.open("w", encoding="utf-8") as file:
        file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {
        "model": model,
        "seed": int(seed),
        "train_size": train_size,
        "holdout_size": holdout_size,
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "timestamp_utc": timestamp,
        "traceback_path": str(traceback_path),
    }


def _resource_limit_failure(
    benchmark_dir: Path,
    model: str,
    seed: int,
    train_size: int | None,
    holdout_size: int | None,
    summary_row: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any] | None:
    reasons: list[str] = []
    training_limit = limits.get("max_training_seconds_per_run")
    total_limit = limits.get("max_total_seconds_per_run")
    memory_limit = limits.get("max_peak_memory_mb")
    if training_limit is not None and summary_row.get("training_seconds") is not None and float(summary_row["training_seconds"]) > float(training_limit):
        reasons.append(f"training_seconds={summary_row['training_seconds']} exceeded limit {training_limit}")
    if total_limit is not None and summary_row.get("duration_seconds") is not None and float(summary_row["duration_seconds"]) > float(total_limit):
        reasons.append(f"duration_seconds={summary_row['duration_seconds']} exceeded limit {total_limit}")
    if memory_limit is not None and summary_row.get("peak_memory_mb") is not None and float(summary_row["peak_memory_mb"]) > float(memory_limit):
        reasons.append(f"peak_memory_mb={summary_row['peak_memory_mb']} exceeded limit {memory_limit}")
    if not reasons:
        return None
    exc = ResourceLimitExceeded("; ".join(reasons))
    failure = _record_failure(
        benchmark_dir=benchmark_dir,
        model=model,
        seed=seed,
        stage="resource_limits",
        exc=exc,
        train_size=train_size,
        holdout_size=holdout_size,
    )
    failure["status"] = "resource_limited"
    failure["reasons"] = reasons
    return failure


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
    aggregate_by_model_and_size = aggregate_summary_by_model_and_size(summary_rows, failures, config["benchmark"]["models"])
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
        aggregate_by_model_and_size=aggregate_by_model_and_size,
        marginal_gains=calculate_marginal_gains(summary_rows),
        scalability_limits=calculate_scalability_limits(summary_rows, failures, config["benchmark"]["models"], config["benchmark"].get("train_sizes", [])),
        warmup_seconds={},
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
    aggregate_by_model_and_size: dict[str, Any],
    marginal_gains: list[dict[str, Any]],
    scalability_limits: list[dict[str, Any]],
    warmup_seconds: dict[str, float],
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
        "aggregate_by_model_and_size": aggregate_by_model_and_size,
        "marginal_gains_count": len(marginal_gains),
        "scalability_limits": scalability_limits,
        "interpretation": "Exploratory pilot benchmark; do not treat a single metric as a final model ranking.",
    }
    if export_json:
        outputs["summary_json"] = write_json(summary_payload, benchmark_dir / "summary.json")
        outputs["runs_json"] = write_json({"runs": run_references}, benchmark_dir / "runs.json")
        outputs["failures_json"] = write_json({"failures": failures}, benchmark_dir / "failures.json")
        outputs["environment_json"] = write_json(_benchmark_environment(warmup_seconds), benchmark_dir / "environment.json")
        outputs["resource_limits_json"] = write_json({"resource_limits": config.get("resource_limits", {})}, benchmark_dir / "resource_limits.json")
        outputs["aggregate_by_model_and_size_json"] = write_json(
            {"aggregate_by_model_and_size": aggregate_by_model_and_size},
            benchmark_dir / "aggregate_by_model_and_size.json",
        )
        outputs["marginal_gains_json"] = write_json({"marginal_gains": marginal_gains}, benchmark_dir / "marginal_gains.json")
        outputs["scalability_limits_json"] = write_json({"scalability_limits": scalability_limits}, benchmark_dir / "scalability_limits.json")
        outputs["plot_data_json"] = write_json({"rows": summary_rows}, benchmark_dir / "plot_data.json")
    if export_parquet:
        outputs["plot_data_parquet"] = benchmark_dir / "plot_data.parquet"
        summary_frame.to_parquet(outputs["plot_data_parquet"], index=False)
    if export_csv:
        outputs["plot_data_csv"] = benchmark_dir / "plot_data.csv"
        summary_frame.to_csv(outputs["plot_data_csv"], index=False)

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
            "same_calibration_per_seed_and_train_size": "train_sizes" in config["benchmark"],
            "same_train_holdout_per_seed_and_train_size": "train_sizes" in config["benchmark"],
            "assessment_mode": config["benchmark"]["assessment_mode"],
            "parallelism": config.get("execution", {}).get("parallelism", 1),
            "warmup_backends": config.get("execution", {}).get("warmup_backends", False),
            "rotate_model_order_by_seed": config.get("execution", {}).get("rotate_model_order_by_seed", False),
            "train_sizes": config["benchmark"].get("train_sizes"),
            "calibration_rows": config["benchmark"].get("calibration_rows"),
            "holdout_fraction": config["benchmark"]["holdout_fraction"],
        },
    }


def _benchmark_environment(warmup_seconds: dict[str, float]) -> dict[str, Any]:
    info = environment_info()
    info["resource_monitor"] = {
        "psutil_available": _psutil_available(),
        "memory_unit": "MiB",
        "cpu_count": os.cpu_count(),
    }
    info["backend_warmup_seconds"] = warmup_seconds
    return info


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True
