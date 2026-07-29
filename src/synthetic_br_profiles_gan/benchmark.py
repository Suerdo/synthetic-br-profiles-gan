"""Orquestração de benchmarks para comparar sintetizadores tabulares."""

from __future__ import annotations

import logging
import gc
import json
import math
import os
import re
import subprocess
import sys
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
from synthetic_br_profiles_gan.evaluation.income_calibration import (
    REQUIRED_INCOME_OCCUPATIONS,
    run_income_calibration_analysis,
)
from synthetic_br_profiles_gan.evaluation.vocabulary import (
    DEFAULT_LOW_COUNT_THRESHOLD,
    DEFAULT_MINIMUM_INCOME_GROUP_COUNT,
    DEFAULT_RARE_OCCUPATION_THRESHOLD,
    evaluate_vocabulary_v2_quality,
)
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
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed

LOGGER = logging.getLogger(__name__)


class ResourceLimitExceeded(PipelineError):
    """Gerada quando um limite de recurso configurado para benchmark é excedido."""

DEFAULT_BENCHMARK_CONFIG: ConfigDict = {
    "benchmark": {
        "name": "pilot",
        "type": "quality",
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
    "execution": {
        "parallelism": 1,
        "warmup_backends": False,
        "rotate_model_order_by_seed": False,
        "subprocess_isolation": False,
        "subprocess_timeout_seconds": None,
    },
    "resource_limits": {
        "max_training_seconds_per_run": None,
        "max_total_seconds_per_run": None,
        "max_peak_memory_mb": None,
        "stop_larger_sizes_after_resource_failure": True,
    },
    "vocabulary_quality": {
        "rare_occupation_threshold": DEFAULT_RARE_OCCUPATION_THRESHOLD,
        "minimum_income_group_count": DEFAULT_MINIMUM_INCOME_GROUP_COUNT,
        "low_count_threshold": DEFAULT_LOW_COUNT_THRESHOLD,
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
    "duplicate_base_row_rate",
    "duplicate_base_duplicated_occurrences",
    "duplicate_base_duplicated_groups",
    "duplicate_base_largest_group",
    "unique_combination_rate",
    "unique_combinations",
    "exact_train_match_rate",
    "exact_train_match_count",
    "exact_holdout_match_rate",
    "exact_holdout_match_count",
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
    "occupation_raw_coverage",
    "occupation_final_coverage",
    "occupation_distribution_distance_raw",
    "occupation_distribution_distance_final",
    "occupation_entropy_raw",
    "occupation_entropy_final",
    "most_frequent_occupation_share_raw",
    "most_frequent_occupation_share_final",
    "education_occupation_valid_rate_raw",
    "education_occupation_valid_rate_final",
    "age_occupation_valid_rate_raw",
    "age_occupation_valid_rate_final",
    "legacy_occupations_raw_count",
    "legacy_occupations_final_count",
    "unicode_nfc_valid_raw",
    "unicode_nfc_valid_final",
    "vocabulary_quality_gate_status",
]
CAPACITY_RESULT_COLUMNS = [
    "benchmark_id",
    "model",
    "seed",
    "train_size",
    "holdout_size",
    "calibration_rows",
    "status",
    "quality_status",
    "run_id",
    "failure_stage",
    "failure_type",
    "failure_message",
    "exit_code",
    "exit_signal",
    "duration_seconds",
    "training_seconds",
    "generation_seconds",
    "validation_seconds",
    "evaluation_seconds",
    "export_seconds",
    "memory_initial_mb",
    "peak_memory_mb",
    "memory_incremental_mb",
    "model_size_mb",
    "artifact_size_mb",
    "backend_warmup_seconds",
    "batches_per_epoch",
    "generator_updates",
    "discriminator_updates",
    "mean_epoch_seconds",
    "ctgan_batches_per_epoch_inferred",
    "ctgan_total_batches_inferred",
    "backend",
    "cpu_count",
    "gpu",
    "library_versions",
    "python_version",
    "platform",
    "backend_version",
    "stdout_log",
    "stderr_log",
    "result_json",
    "result_json_available",
    "last_worker_event",
    "timestamp_utc",
]
CAPACITY_COMPLETED_STATUSES = {"completed", "quality_quarantined", "quality_rejected"}


def build_benchmark_id(name: str, timestamp: datetime | None = None) -> str:
    """Cria um benchmark id com prefixo estável, timestamp e sufixo curto."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip().lower()).strip("-") or "benchmark"
    return f"{slug}-{build_run_id(timestamp)}"


def resolve_benchmark_config(config: ConfigDict | None = None) -> ConfigDict:
    """Mescla padrões do benchmark e valida a configuração resolvida."""
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
    """Retorna a matriz configurada de modelo x seed x train_size opcional."""
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
    """Retorna o tamanho exato de holdout implícito por um tamanho-alvo de treino."""
    value = int(train_size) * float(holdout_fraction) / (1.0 - float(holdout_fraction))
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_size and holdout_fraction do not produce an exact integer holdout size.")
    return rounded


def train_size_specs(config: ConfigDict) -> list[dict[str, int | None]]:
    """Retorna as especificações configuradas de train_size."""
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
    """Alterna a ordem de execução dos modelos por seed para reduzir viés de cache e warm-up."""
    if not rotate or not models:
        return list(models)
    offset = seed_index % len(models)
    return [*models[offset:], *models[:offset]]


class ResourceMonitor:
    """Amostra memória residente do processo durante uma execução de benchmark."""

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
        """Retorna a memória residente atual em MiB quando psutil está disponível."""
        if self._process is None:
            return None
        return float(self._process.memory_info().rss / (1024 * 1024))

    def thread_count(self) -> int | None:
        """Retorna a quantidade atual de threads do processo quando psutil está disponível."""
        if self._process is None:
            return None
        return int(self._process.num_threads())

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = self.current_memory_mb()
            if current is not None:
                self.peak_memory_mb = max(current, self.peak_memory_mb or current)


def run_capacity_benchmark(
    config: ConfigDict,
    started_at: datetime | None = None,
    started_perf: float | None = None,
) -> dict[str, Any]:
    """Executa um benchmark de capacidade operacional com subprocessos isolados por modelo."""
    started = started_at or datetime.now(timezone.utc)
    perf_start = started_perf or time.perf_counter()
    benchmark_id = build_benchmark_id(str(config["benchmark"]["name"]), started)
    benchmark_dir = Path(config["outputs"]["base_directory"]) / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, benchmark_dir / "benchmark_config.yaml")
    LOGGER.info("capacity_benchmark_started", extra={"benchmark_id": benchmark_id})

    metadata = default_metadata()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_references: list[dict[str, Any]] = []
    stop_after_failure: dict[str, int] = {}
    continue_on_error = bool(config["benchmark"].get("continue_on_error", True))

    for seed in config["benchmark"]["seeds"]:
        seed = int(seed)
        for size_spec in train_size_specs(config):
            train_size = int(size_spec["train_size"])
            holdout_size = int(size_spec["holdout_size"])
            calibration_rows = int(size_spec["calibration_rows"])
            active_models = [
                model
                for model in config["benchmark"]["models"]
                if model not in stop_after_failure or train_size <= stop_after_failure[model]
            ]
            for model in config["benchmark"]["models"]:
                if model in stop_after_failure and train_size > stop_after_failure[model]:
                    row = _capacity_skipped_row(benchmark_id, model, seed, train_size, holdout_size, calibration_rows, stop_after_failure[model])
                    rows.append(row)
                    failure = _capacity_failure_from_row(row, "progression", "SkippedAfterFailure", row["failure_type"])
                    failures.append(failure)
                    run_references.append(_capacity_run_reference(row))
            if not active_models:
                continue

            try:
                calibration_started = time.perf_counter()
                calibration_config = _calibration_config_for_seed(config, seed, calibration_rows=calibration_rows)
                calibration = generate_calibration_dataset(config=calibration_config)
                calibration_seconds = float(time.perf_counter() - calibration_started)
                split_started = time.perf_counter()
                train, holdout = split_train_holdout(
                    calibration,
                    holdout_fraction=float(config["benchmark"]["holdout_fraction"]),
                    seed=seed,
                    train_rows=train_size,
                    holdout_rows=holdout_size,
                )
                split_seconds = float(time.perf_counter() - split_started)
                split_dir = _calibration_output_dir(benchmark_dir, seed, train_size)
                split_paths = save_calibration_splits(calibration, train, holdout, split_dir, metadata=metadata)
            except Exception as exc:
                _record_failure(benchmark_dir, "all", seed, "calibration", exc, train_size=train_size, holdout_size=holdout_size)
                for model in active_models:
                    row = _capacity_error_row(
                        benchmark_id=benchmark_id,
                        model=model,
                        seed=seed,
                        train_size=train_size,
                        holdout_size=holdout_size,
                        calibration_rows=calibration_rows,
                        failure_stage="calibration",
                        failure_type=type(exc).__name__,
                        message=str(exc),
                    )
                    rows.append(row)
                    failures.append(_capacity_failure_from_row(row, "calibration", type(exc).__name__, str(exc)))
                    run_references.append(_capacity_run_reference(row))
                    if bool(config.get("resource_limits", {}).get("stop_larger_sizes_after_resource_failure", True)):
                        stop_after_failure.setdefault(model, train_size)
                if not continue_on_error:
                    raise
                continue
            finally:
                try:
                    del calibration
                    del train
                    del holdout
                except UnboundLocalError:
                    pass
                gc.collect()

            calibration_timings = {"calibration_seconds": calibration_seconds, "split_seconds": split_seconds}
            for model in active_models:
                row = _run_capacity_model_subprocess(
                    config=config,
                    benchmark_id=benchmark_id,
                    benchmark_dir=benchmark_dir,
                    model=model,
                    seed=seed,
                    train_size=train_size,
                    holdout_size=holdout_size,
                    calibration_rows=calibration_rows,
                    train_path=split_paths["train"],
                    holdout_path=split_paths["holdout"],
                    metadata_path=split_paths["metadata"],
                    calibration_timings=calibration_timings,
                )
                rows.append(row)
                run_references.append(_capacity_run_reference(row))
                if row["status"] not in CAPACITY_COMPLETED_STATUSES:
                    stop_after_failure.setdefault(model, train_size)
                    failures.append(_capacity_failure_from_row(row, row.get("failure_stage") or "subprocess", row.get("failure_type") or row["status"], row["status"]))
                    if not continue_on_error:
                        ended = datetime.now(timezone.utc)
                        outputs = _write_capacity_outputs(
                            benchmark_dir,
                            config,
                            benchmark_id,
                            started,
                            ended,
                            float(time.perf_counter() - perf_start),
                            rows,
                            failures,
                            run_references,
                        )
                        raise PipelineError(f"Capacity benchmark stopped after {model} train_size={train_size}: {row['status']}")

    ended = datetime.now(timezone.utc)
    completed_runs = sum(1 for row in rows if row["status"] in CAPACITY_COMPLETED_STATUSES)
    failed_runs = sum(1 for row in rows if row["status"] not in CAPACITY_COMPLETED_STATUSES)
    overall_status = "completed" if failed_runs == 0 else "completed_with_failures" if completed_runs else "failed"
    outputs = _write_capacity_outputs(
        benchmark_dir=benchmark_dir,
        config=config,
        benchmark_id=benchmark_id,
        started_at=started,
        ended_at=ended,
        duration_seconds=float(time.perf_counter() - perf_start),
        rows=rows,
        failures=failures,
        run_references=run_references,
    )
    LOGGER.info(
        "capacity_benchmark_finished",
        extra={"benchmark_id": benchmark_id, "status": overall_status, "completed_runs": completed_runs, "failed_runs": failed_runs},
    )
    return {
        "benchmark_id": benchmark_id,
        "status": overall_status,
        "benchmark_dir": benchmark_dir,
        "expected_runs": len(benchmark_matrix(config)),
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "capacity_rows": rows,
        "capacity_summary": calculate_capacity_limits(rows, config["benchmark"]["models"], config["benchmark"].get("train_sizes", [])),
        "failures": failures,
        "paths": outputs,
        "duration_seconds": float((ended - started).total_seconds()),
    }


def _run_capacity_model_subprocess(
    config: ConfigDict,
    benchmark_id: str,
    benchmark_dir: Path,
    model: str,
    seed: int,
    train_size: int,
    holdout_size: int,
    calibration_rows: int,
    train_path: Path,
    holdout_path: Path,
    metadata_path: Path,
    calibration_timings: dict[str, float],
) -> dict[str, Any]:
    subprocess_dir = benchmark_dir / "subprocesses" / model / f"train-{train_size}"
    subprocess_dir.mkdir(parents=True, exist_ok=True)
    run_config = _pipeline_config_for_run(config, model, seed, calibration_rows=calibration_rows)
    run_config_path = save_yaml_config(run_config, subprocess_dir / "run_config.yaml")
    result_path = subprocess_dir / "result.json"
    stdout_path = subprocess_dir / "stdout.log"
    stderr_path = subprocess_dir / "stderr.log"
    command = [
        sys.executable,
        "-m",
        "synthetic_br_profiles_gan.benchmark_worker",
        "--config",
        str(run_config_path),
        "--model",
        model,
        "--train",
        str(train_path),
        "--holdout",
        str(holdout_path),
        "--metadata",
        str(metadata_path),
        "--output",
        str(result_path),
    ]
    if bool(config.get("execution", {}).get("warmup_backends", False)):
        command.append("--warmup-backend")
    execution = run_capacity_subprocess(
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        resource_limits=config.get("resource_limits", {}),
        timeout_seconds=config.get("execution", {}).get("subprocess_timeout_seconds"),
    )
    payload = _load_capacity_worker_result(result_path)
    row = _capacity_row_from_worker_payload(
        benchmark_id=benchmark_id,
        model=model,
        seed=seed,
        train_size=train_size,
        holdout_size=holdout_size,
        calibration_rows=calibration_rows,
        calibration_timings=calibration_timings,
        payload=payload,
        execution=execution,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )
    if row["status"] in CAPACITY_COMPLETED_STATUSES:
        limit_failure = _capacity_limit_failure(row, config.get("resource_limits", {}))
        if limit_failure is not None:
            row.update(limit_failure)
    return row


def run_capacity_subprocess(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    resource_limits: dict[str, Any] | None = None,
    timeout_seconds: float | int | None = None,
    poll_interval_seconds: float = 0.2,
) -> dict[str, Any]:
    """Executa um worker de capacidade e monitora a memória residente da árvore de processos."""
    limits = resource_limits or {}
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    memory_initial: float | None = None
    memory_peak: float | None = None
    limit_reason: str | None = None
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True)
        while True:
            if process.poll() is not None:
                break
            memory = _process_tree_rss_mb(process.pid)
            if memory is not None:
                if memory_initial is None:
                    memory_initial = memory
                memory_peak = max(memory_peak or memory, memory)
            elapsed = time.perf_counter() - started
            max_total = _limit_value(limits.get("max_total_seconds_per_run"))
            max_memory = _limit_value(limits.get("max_peak_memory_mb"))
            timeout_limit = _limit_value(timeout_seconds)
            if max_memory is not None and memory_peak is not None and memory_peak > max_memory:
                limit_reason = f"peak_memory_mb={memory_peak:.3f} exceeded limit {max_memory:.3f}"
                _terminate_process_tree(process)
                break
            if max_total is not None and elapsed > max_total:
                limit_reason = f"duration_seconds={elapsed:.3f} exceeded limit {max_total:.3f}"
                _terminate_process_tree(process)
                break
            if timeout_limit is not None and elapsed > timeout_limit:
                timed_out = True
                limit_reason = f"subprocess_timeout_seconds={elapsed:.3f} exceeded limit {timeout_limit:.3f}"
                _terminate_process_tree(process)
                break
            time.sleep(float(poll_interval_seconds))
        exit_code = process.returncode
        if exit_code is None:
            exit_code = process.wait()
    duration = float(time.perf_counter() - started)
    if memory_peak is None:
        memory_peak = memory_initial
    return {
        "exit_code": int(exit_code),
        "duration_seconds": duration,
        "memory_initial_mb": memory_initial,
        "peak_memory_mb": memory_peak,
        "memory_incremental_mb": None if memory_initial is None or memory_peak is None else float(memory_peak - memory_initial),
        "resource_limited": limit_reason is not None,
        "limit_reason": limit_reason,
        "timed_out": timed_out,
    }


def _process_tree_rss_mb(pid: int) -> float | None:
    try:
        import psutil

        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
        total = 0
        for item in processes:
            try:
                total += item.memory_info().rss
            except psutil.Error:
                continue
        return float(total / (1024 * 1024))
    except Exception:
        return None


def _terminate_process_tree(process: subprocess.Popen) -> None:
    children = []
    try:
        import psutil

        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                continue
        _terminate_parent_with_popen(process)
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                continue
        if alive:
            psutil.wait_procs(alive, timeout=5)
    except Exception:
        _terminate_parent_with_popen(process)


def _terminate_parent_with_popen(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            process.terminate()
        except (ProcessLookupError, OSError):
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
        process.wait(timeout=5)


def _limit_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_capacity_worker_result(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        with result_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _capacity_row_from_worker_payload(
    benchmark_id: str,
    model: str,
    seed: int,
    train_size: int,
    holdout_size: int,
    calibration_rows: int,
    calibration_timings: dict[str, float],
    payload: dict[str, Any] | None,
    execution: dict[str, Any],
    stdout_path: Path,
    stderr_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    result_json_available = result_path.exists()
    if execution.get("resource_limited"):
        status = "resource_limited"
        failure_stage = "resource_monitor"
        failure_type = "ResourceLimitExceeded"
        quality_status = None
    elif payload is None:
        status = "failed"
        failure_stage = "subprocess"
        failure_type = "MissingWorkerResult"
        quality_status = None
    else:
        status = str(payload.get("technical_status") or "failed")
        failure_stage = payload.get("failure_stage")
        failure_type = payload.get("failure_type")
        quality_status = payload.get("quality_status")
        if status == "failed" and _is_resource_exhaustion_failure(failure_type, payload.get("message")):
            status = "resource_limited"

    stage = payload.get("stage_durations", {}) if payload else {}
    env = payload.get("environment", {}) if payload else environment_info()
    versions = env.get("library_versions", {}) if isinstance(env, dict) else {}
    gpu = env.get("gpu", {}) if isinstance(env, dict) else {}
    exit_code = execution.get("exit_code")
    exit_signal = -int(exit_code) if isinstance(exit_code, int) and exit_code < 0 else None
    failure_message = _capacity_failure_message(payload, execution, result_json_available)
    row = {
        "benchmark_id": benchmark_id,
        "model": model,
        "seed": int(seed),
        "train_size": int(train_size),
        "holdout_size": int(holdout_size),
        "calibration_rows": int(calibration_rows),
        "status": status,
        "quality_status": quality_status,
        "run_id": payload.get("run_id") if payload else None,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "exit_code": exit_code,
        "exit_signal": exit_signal,
        "duration_seconds": payload.get("duration_seconds") if payload and payload.get("duration_seconds") is not None else execution.get("duration_seconds"),
        "calibration_seconds": calibration_timings.get("calibration_seconds"),
        "split_seconds": calibration_timings.get("split_seconds"),
        "training_seconds": stage.get("training_seconds"),
        "generation_seconds": stage.get("generation_seconds"),
        "validation_seconds": stage.get("validation_seconds"),
        "evaluation_seconds": stage.get("evaluation_seconds"),
        "export_seconds": stage.get("export_seconds"),
        "memory_initial_mb": execution.get("memory_initial_mb"),
        "peak_memory_mb": execution.get("peak_memory_mb"),
        "memory_incremental_mb": execution.get("memory_incremental_mb"),
        "model_size_mb": payload.get("model_size_mb") if payload else None,
        "artifact_size_mb": payload.get("artifact_size_mb") if payload else None,
        "backend_warmup_seconds": payload.get("backend_warmup_seconds") if payload else None,
        "batches_per_epoch": payload.get("batches_per_epoch") if payload else None,
        "generator_updates": payload.get("generator_updates") if payload else None,
        "discriminator_updates": payload.get("discriminator_updates") if payload else None,
        "mean_epoch_seconds": payload.get("mean_epoch_seconds") if payload else None,
        "ctgan_batches_per_epoch_inferred": payload.get("ctgan_batches_per_epoch_inferred") if payload else None,
        "ctgan_total_batches_inferred": payload.get("ctgan_total_batches_inferred") if payload else None,
        "backend": payload.get("backend") if payload else model,
        "cpu_count": os.cpu_count(),
        "gpu": json.dumps(gpu, ensure_ascii=False, sort_keys=True, default=str),
        "library_versions": json.dumps(versions, ensure_ascii=False, sort_keys=True, default=str),
        "python_version": env.get("python_version") if isinstance(env, dict) else None,
        "platform": env.get("platform") if isinstance(env, dict) else None,
        "backend_version": _backend_version(model, versions),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "result_json": str(result_path) if result_json_available else None,
        "result_json_available": bool(result_json_available),
        "last_worker_event": _last_worker_event(stdout_path, stderr_path),
        "timestamp_utc": payload.get("timestamp_utc") if payload else datetime.now(timezone.utc).isoformat(),
        "limit_reason": execution.get("limit_reason"),
    }
    if status == "failed" and row["failure_type"] is None:
        row["failure_type"] = "WorkerFailed"
    if status == "resource_limited" and row["failure_type"] is None:
        row["failure_type"] = "ResourceLimitExceeded"
    return row


def _capacity_failure_message(payload: dict[str, Any] | None, execution: dict[str, Any], result_json_available: bool) -> str | None:
    if payload and payload.get("message"):
        return str(payload["message"])
    if execution.get("limit_reason"):
        return str(execution["limit_reason"])
    if not result_json_available:
        return (
            "O subprocesso foi encerrado antes de produzir result.json. "
            "Os registros disponíveis não permitem determinar com segurança a causa raiz."
        )
    return None


def _is_resource_exhaustion_failure(failure_type: Any, message: Any) -> bool:
    normalized_type = str(failure_type or "")
    normalized_message = str(message or "").casefold()
    if normalized_type == "MemoryError":
        return True
    if normalized_type != "OSError":
        return False
    resource_fragments = [
        "winerror 1450",
        "recursos de sistema suficientes",
        "insufficient system resources",
        "not enough memory",
        "cannot allocate memory",
        "out of memory",
        "resource exhausted",
    ]
    return any(fragment in normalized_message for fragment in resource_fragments)


def _backend_version(model: str, versions: dict[str, Any]) -> str | None:
    if not isinstance(versions, dict):
        return None
    if model == "ctgan":
        return versions.get("ctgan")
    if model == "simple_gan":
        return versions.get("tensorflow")
    return None


def _last_worker_event(stdout_path: Path, stderr_path: Path) -> str | None:
    for path in (stderr_path, stdout_path):
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError:
            continue
        if lines:
            return lines[-1]
    return None


def _capacity_limit_failure(row: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any] | None:
    training_limit = _limit_value(limits.get("max_training_seconds_per_run"))
    if training_limit is not None and row.get("training_seconds") is not None and float(row["training_seconds"]) > training_limit:
        return {
            "status": "resource_limited",
            "failure_stage": "treinamento",
            "failure_type": "ResourceLimitExceeded",
            "limit_reason": f"training_seconds={float(row['training_seconds']):.3f} exceeded limit {training_limit:.3f}",
        }
    return None


def _capacity_skipped_row(
    benchmark_id: str,
    model: str,
    seed: int,
    train_size: int,
    holdout_size: int,
    calibration_rows: int,
    failed_train_size: int,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "model": model,
        "seed": int(seed),
        "train_size": int(train_size),
        "holdout_size": int(holdout_size),
        "calibration_rows": int(calibration_rows),
        "status": "skipped_after_failure",
        "quality_status": None,
        "run_id": None,
        "failure_stage": "progression",
        "failure_type": f"Skipped after failure at train_size={failed_train_size}",
        "failure_message": f"Tamanho pulado porque houve falha anterior em train_size={failed_train_size}.",
        "exit_code": None,
        "exit_signal": None,
        "duration_seconds": 0.0,
        "training_seconds": None,
        "generation_seconds": None,
        "validation_seconds": None,
        "evaluation_seconds": None,
        "export_seconds": None,
        "memory_initial_mb": None,
        "peak_memory_mb": None,
        "memory_incremental_mb": None,
        "model_size_mb": None,
        "artifact_size_mb": None,
        "backend_warmup_seconds": None,
        "batches_per_epoch": None,
        "generator_updates": None,
        "discriminator_updates": None,
        "mean_epoch_seconds": None,
        "ctgan_batches_per_epoch_inferred": None,
        "ctgan_total_batches_inferred": None,
        "backend": model,
        "cpu_count": os.cpu_count(),
        "gpu": None,
        "library_versions": None,
        "python_version": sys.version,
        "platform": sys.platform,
        "backend_version": None,
        "stdout_log": None,
        "stderr_log": None,
        "result_json": None,
        "result_json_available": False,
        "last_worker_event": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limit_reason": None,
    }


def _capacity_error_row(
    benchmark_id: str,
    model: str,
    seed: int,
    train_size: int,
    holdout_size: int,
    calibration_rows: int,
    failure_stage: str,
    failure_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "model": model,
        "seed": int(seed),
        "train_size": int(train_size),
        "holdout_size": int(holdout_size),
        "calibration_rows": int(calibration_rows),
        "status": "failed",
        "quality_status": None,
        "run_id": None,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
        "failure_message": message,
        "exit_code": None,
        "exit_signal": None,
        "duration_seconds": 0.0,
        "training_seconds": None,
        "generation_seconds": None,
        "validation_seconds": None,
        "evaluation_seconds": None,
        "export_seconds": None,
        "memory_initial_mb": None,
        "peak_memory_mb": None,
        "memory_incremental_mb": None,
        "model_size_mb": None,
        "artifact_size_mb": None,
        "backend_warmup_seconds": None,
        "batches_per_epoch": None,
        "generator_updates": None,
        "discriminator_updates": None,
        "mean_epoch_seconds": None,
        "ctgan_batches_per_epoch_inferred": None,
        "ctgan_total_batches_inferred": None,
        "backend": model,
        "cpu_count": os.cpu_count(),
        "gpu": None,
        "library_versions": None,
        "python_version": sys.version,
        "platform": sys.platform,
        "backend_version": None,
        "stdout_log": None,
        "stderr_log": None,
        "result_json": None,
        "result_json_available": False,
        "last_worker_event": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limit_reason": None,
    }


def _capacity_failure_from_row(row: dict[str, Any], stage: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "model": row["model"],
        "seed": int(row["seed"]),
        "train_size": row.get("train_size"),
        "holdout_size": row.get("holdout_size"),
        "calibration_rows": row.get("calibration_rows"),
        "stage": stage,
        "failure_stage": row.get("failure_stage") or stage,
        "status": row["status"],
        "error_type": error_type,
        "failure_type": row.get("failure_type") or error_type,
        "message": message,
        "failure_message": row.get("failure_message") or message,
        "exit_code": row.get("exit_code"),
        "exit_signal": row.get("exit_signal"),
        "duration_seconds": row.get("duration_seconds"),
        "memory_initial_mb": row.get("memory_initial_mb"),
        "peak_memory_mb": row.get("peak_memory_mb"),
        "memory_incremental_mb": row.get("memory_incremental_mb"),
        "stdout_log": row.get("stdout_log"),
        "stderr_log": row.get("stderr_log"),
        "result_json": row.get("result_json"),
        "result_json_available": bool(row.get("result_json_available")),
        "last_worker_event": row.get("last_worker_event"),
        "python_version": row.get("python_version"),
        "platform": row.get("platform"),
        "backend": row.get("backend"),
        "backend_version": row.get("backend_version"),
        "cpu_count": row.get("cpu_count"),
        "gpu": row.get("gpu"),
        "library_versions": row.get("library_versions"),
        "limit_reason": row.get("limit_reason"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _capacity_run_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": row["benchmark_id"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "train_size": row.get("train_size"),
        "holdout_size": row.get("holdout_size"),
        "status": row["status"],
        "quality_status": row.get("quality_status"),
        "run_id": row.get("run_id"),
        "stdout_log": row.get("stdout_log"),
        "stderr_log": row.get("stderr_log"),
        "result_json": row.get("result_json"),
    }


def calculate_capacity_limits(rows: list[dict[str, Any]], models: list[str], train_sizes: list[int]) -> list[dict[str, Any]]:
    """Calcula limites de capacidade observados por modelo sem afirmar limites absolutos."""
    sorted_sizes = sorted(int(size) for size in train_sizes)
    payload: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model and row.get("train_size") is not None]
        completed = sorted(
            set(
                int(row["train_size"])
                for row in model_rows
                if row["status"] in CAPACITY_COMPLETED_STATUSES
            )
        )
        non_completed = sorted(
            (
                row
                for row in model_rows
                if row["status"] not in CAPACITY_COMPLETED_STATUSES and row["status"] != "skipped_after_failure"
            ),
            key=lambda row: int(row["train_size"]),
        )
        skipped = sorted(
            set(
                int(row["train_size"])
                for row in model_rows
                if row["status"] == "skipped_after_failure"
            )
        )
        first_failure = non_completed[0] if non_completed else None
        first_failed_size = int(first_failure["train_size"]) if first_failure is not None else None
        first_failure_status = str(first_failure["status"]) if first_failure is not None else None
        first_failure_type = str(first_failure.get("failure_type") or first_failure_status) if first_failure is not None else None
        largest = max(completed) if completed else None
        if largest is None:
            conclusion = "Nenhum tamanho foi concluído com sucesso neste ambiente. O limite máximo absoluto não foi determinado."
        elif first_failed_size is None:
            conclusion = (
                f"O modelo foi executado com sucesso com pelo menos {largest:,} registros neste ambiente. "
                "O limite máximo absoluto não foi determinado."
            ).replace(",", ".")
        else:
            conclusion = (
                f"O modelo foi executado com sucesso até {largest:,} registros neste ambiente. "
                f"A primeira falha foi observada em {first_failed_size:,} registros. "
                "O limite máximo absoluto não foi determinado."
            ).replace(",", ".")
        tested = sorted(set(int(row["train_size"]) for row in model_rows if row["status"] != "skipped_after_failure"))
        payload.append(
            {
                "model": model,
                "configured_train_sizes": sorted_sizes,
                "tested_train_sizes": tested,
                "completed_train_sizes": completed,
                "largest_tested_successful_size": largest,
                "first_failed_size": first_failed_size,
                "first_failure_status": first_failure_status,
                "first_failure_type": first_failure_type,
                "skipped_train_sizes": skipped,
                "untested_larger_sizes": skipped,
                "observed_interval_lower": largest,
                "observed_interval_upper": first_failed_size,
                "conclusion": conclusion,
            }
        )
    return payload


def _write_capacity_outputs(
    benchmark_dir: Path,
    config: ConfigDict,
    benchmark_id: str,
    started_at: datetime,
    ended_at: datetime,
    duration_seconds: float,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_references: list[dict[str, Any]],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    frame = pd.DataFrame(rows, columns=[*CAPACITY_RESULT_COLUMNS, "calibration_seconds", "split_seconds", "limit_reason"])
    export_csv = bool(config["outputs"].get("export_csv", True))
    export_parquet = bool(config["outputs"].get("export_parquet", True))
    export_json = bool(config["outputs"].get("export_json", True))
    if export_parquet:
        outputs["capacity_results_parquet"] = benchmark_dir / "capacity_results.parquet"
        frame.to_parquet(outputs["capacity_results_parquet"], index=False)
    if export_csv:
        outputs["capacity_results_csv"] = benchmark_dir / "capacity_results.csv"
        frame.to_csv(outputs["capacity_results_csv"], index=False)

    capacity_limits = calculate_capacity_limits(rows, config["benchmark"]["models"], config["benchmark"].get("train_sizes", []))
    summary = {
        "benchmark_id": benchmark_id,
        "status": "completed" if not failures else "completed_with_failures" if rows else "failed",
        "expected_runs": len(benchmark_matrix(config)),
        "completed_runs": sum(1 for row in rows if row["status"] in CAPACITY_COMPLETED_STATUSES),
        "failed_runs": sum(1 for row in rows if row["status"] not in CAPACITY_COMPLETED_STATUSES),
        "capacity_limits": capacity_limits,
        "interpretation": "Operational capacity benchmark; quality gates do not determine technical capacity.",
    }
    if export_json:
        outputs["capacity_summary_json"] = write_json(summary, benchmark_dir / "capacity_summary.json")
        outputs["failures_json"] = write_json({"failures": failures}, benchmark_dir / "failures.json")
        outputs["runs_json"] = write_json({"runs": run_references}, benchmark_dir / "runs.json")
        outputs["scalability_limits_json"] = write_json({"scalability_limits": capacity_limits}, benchmark_dir / "scalability_limits.json")

    manifest = _benchmark_manifest(
        benchmark_id=benchmark_id,
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        expected_runs=len(benchmark_matrix(config)),
        completed_runs=summary["completed_runs"],
        failed_runs=summary["failed_runs"],
        status=summary["status"],
        artifact_paths={**outputs, "benchmark_config": benchmark_dir / "benchmark_config.yaml"},
    )
    outputs["benchmark_manifest"] = write_json(manifest, benchmark_dir / "benchmark_manifest.json")
    return outputs


def run_income_calibration_benchmark(
    config: ConfigDict,
    started_at: datetime | None = None,
    started_perf: float | None = None,
) -> dict[str, Any]:
    """Executa avaliação controlada de versões do modelo sintético de renda."""
    started = started_at or datetime.now(timezone.utc)
    perf_started = started_perf or time.perf_counter()
    benchmark_id = build_benchmark_id(str(config["benchmark"]["name"]), started)
    benchmark_dir = Path(config["outputs"]["base_directory"]) / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, benchmark_dir / "benchmark_config.yaml")
    calibration_config = config.get("income_calibration", {})
    rows_per_occupation = int(calibration_config.get("rows_per_occupation", 5000))
    occupations = tuple(calibration_config.get("occupations", REQUIRED_INCOME_OCCUPATIONS))
    analysis = run_income_calibration_analysis(
        seeds=[int(seed) for seed in config["benchmark"]["seeds"]],
        rows_per_occupation=rows_per_occupation,
        occupations=occupations,
    )
    summary_frame = pd.DataFrame(analysis["rows"])
    compression_frame = pd.DataFrame(analysis["compression"])
    overlap_frame = pd.DataFrame(analysis["overlap"])
    ranking_frame = pd.DataFrame(analysis["ranking"])
    outputs: dict[str, Path] = {
        "summary_csv": benchmark_dir / "income_calibration_summary.csv",
        "summary_parquet": benchmark_dir / "income_calibration_summary.parquet",
        "compression_csv": benchmark_dir / "income_calibration_compression.csv",
        "overlap_csv": benchmark_dir / "income_calibration_overlap.csv",
        "ranking_csv": benchmark_dir / "income_calibration_ranking.csv",
    }
    summary_frame.to_csv(outputs["summary_csv"], index=False, encoding="utf-8-sig")
    summary_frame.to_parquet(outputs["summary_parquet"], index=False)
    compression_frame.to_csv(outputs["compression_csv"], index=False, encoding="utf-8-sig")
    overlap_frame.to_csv(outputs["overlap_csv"], index=False, encoding="utf-8-sig")
    ranking_frame.to_csv(outputs["ranking_csv"], index=False, encoding="utf-8-sig")
    outputs["analysis_json"] = write_json(analysis, benchmark_dir / "income_calibration_analysis.json")
    ended = datetime.now(timezone.utc)
    duration_seconds = float(time.perf_counter() - perf_started)
    outputs["benchmark_manifest"] = write_json(
        _benchmark_manifest(
            benchmark_id=benchmark_id,
            config=config,
            started_at=started,
            ended_at=ended,
            duration_seconds=duration_seconds,
            expected_runs=len(config["benchmark"]["seeds"]) * len(occupations) * 4,
            completed_runs=len(config["benchmark"]["seeds"]) * len(occupations) * 4,
            failed_runs=0,
            status="completed",
            artifact_paths={**outputs, "benchmark_config": benchmark_dir / "benchmark_config.yaml"},
        ),
        benchmark_dir / "benchmark_manifest.json",
    )
    return {
        "benchmark_id": benchmark_id,
        "benchmark_dir": benchmark_dir,
        "status": "completed",
        "completed_runs": len(config["benchmark"]["seeds"]) * len(occupations) * 4,
        "failed_runs": 0,
        "outputs": outputs,
        "summary": {
            "selected_calibration": analysis["selected_calibration"],
            "versions": sorted({row["version_name"] for row in analysis["rows"]}),
            "occupations": list(occupations),
            "seeds": list(config["benchmark"]["seeds"]),
        },
    }


def run_benchmark(config: ConfigDict | None = None) -> dict[str, Any]:
    """Executa o benchmark e grava artefatos no nível do benchmark."""
    started = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    effective = resolve_benchmark_config(config)
    if effective["benchmark"].get("type") == "income_calibration":
        return run_income_calibration_benchmark(effective, started_at=started, started_perf=started_perf)
    if effective["benchmark"].get("type") == "capacity":
        return run_capacity_benchmark(effective, started_at=started, started_perf=started_perf)
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
    """Agrega métricas principais do benchmark por modelo."""
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
    """Agrega métricas principais do benchmark por modelo e tamanho de treino."""
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
    """Calcula mudanças marginais exploratórias entre tamanhos de treino configurados."""
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
    """Resume tamanhos de treino concluídos com sucesso por modelo neste ambiente."""
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
    """Carrega backends opcionais de modelo uma vez e registra o warm-up separadamente."""
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
        if effective["benchmark"].get("type") == "vocabulary_quality":
            raw_sample = _sample_raw_synthesizer_output(
                model=model,
                model_dir=run_result.get("model_dir"),
                rows=int(effective["benchmark"]["synthetic_rows"]),
                seed=seed,
                metadata=metadata,
            )
            vocabulary_config = effective.get("vocabulary_quality", {})
            run_result["vocabulary_quality"] = evaluate_vocabulary_v2_quality(
                reference=holdout,
                raw=raw_sample,
                final=run_result["dataset"],
                metadata=metadata,
                requested_rows=int(effective["benchmark"]["synthetic_rows"]),
                validation_report=run_result.get("validation", {}),
                rare_threshold=float(vocabulary_config.get("rare_occupation_threshold", DEFAULT_RARE_OCCUPATION_THRESHOLD)),
                minimum_income_group_count=int(vocabulary_config.get("minimum_income_group_count", DEFAULT_MINIMUM_INCOME_GROUP_COUNT)),
                low_count_threshold=int(vocabulary_config.get("low_count_threshold", DEFAULT_LOW_COUNT_THRESHOLD)),
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


def _sample_raw_synthesizer_output(
    model: str,
    model_dir: Any,
    rows: int,
    seed: int,
    metadata: DatasetMetadata,
) -> pd.DataFrame:
    """Amostra a saída bruta do sintetizador salvo para diagnóstico de vocabulário."""
    model_path = Path(model_dir)
    set_global_seed(
        int(seed),
        seed_tensorflow=model == "simple_gan",
        seed_torch=model == "ctgan",
    )
    if model == "programmatic":
        from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer

        synthesizer = ProgrammaticSynthesizer.load(model_path)
        return synthesizer.sample(int(rows))
    if model == "simple_gan":
        from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN

        synthesizer = SimpleTabularGAN.load(model_path)
        return synthesizer.sample(int(rows))
    if model == "ctgan":
        from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

        synthesizer = CTGANSynthesizer.load(model_path)
        if getattr(synthesizer, "model", None) is not None:
            sampled = synthesizer.model.sample(int(rows))
            return sampled[[column for column in metadata.model_columns if column in sampled.columns]].copy()
        return synthesizer.sample(int(rows))
    raise ValueError(f"Unsupported model for raw sampling: {model}")


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
    """Retorna uma linha de resumo semilonga para uma execução concluída."""
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
    vocabulary = result.get("vocabulary_quality", {})
    occupation = vocabulary.get("occupation", {})
    coherence = vocabulary.get("coherence", {})
    locale = vocabulary.get("locale", {})
    vocabulary_gates = vocabulary.get("quality_gates", {})
    duplicate_base = privacy.get("duplicate_base_rows") if isinstance(privacy.get("duplicate_base_rows"), dict) else {}
    exact_matches = privacy.get("exact_matches") if isinstance(privacy.get("exact_matches"), dict) else {}
    exact_train = exact_matches.get("train") if isinstance(exact_matches.get("train"), dict) else {}
    exact_holdout = exact_matches.get("holdout") if isinstance(exact_matches.get("holdout"), dict) else {}
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
        "duplicate_base_row_rate": duplicate_base.get("duplicate_row_rate"),
        "duplicate_base_duplicated_occurrences": duplicate_base.get("duplicated_occurrences"),
        "duplicate_base_duplicated_groups": duplicate_base.get("duplicated_groups"),
        "duplicate_base_largest_group": duplicate_base.get("largest_duplicate_group"),
        "unique_combination_rate": privacy.get("unique_combination_rate"),
        "unique_combinations": privacy.get("unique_combinations"),
        "exact_train_match_rate": privacy.get("exact_train_match_rate"),
        "exact_train_match_count": exact_train.get("exact_match_count"),
        "exact_holdout_match_rate": privacy.get("exact_holdout_match_rate"),
        "exact_holdout_match_count": exact_holdout.get("exact_match_count"),
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
        "occupation_raw_coverage": occupation.get("occupation_raw_coverage"),
        "occupation_final_coverage": occupation.get("occupation_final_coverage"),
        "occupation_distribution_distance_raw": occupation.get("occupation_distribution_distance_raw"),
        "occupation_distribution_distance_final": occupation.get("occupation_distribution_distance_final"),
        "occupation_entropy_raw": occupation.get("occupation_entropy_raw"),
        "occupation_entropy_final": occupation.get("occupation_entropy_final"),
        "most_frequent_occupation_share_raw": occupation.get("most_frequent_occupation_share_raw"),
        "most_frequent_occupation_share_final": occupation.get("most_frequent_occupation_share_final"),
        "education_occupation_valid_rate_raw": coherence.get("education_occupation_valid_rate_raw"),
        "education_occupation_valid_rate_final": coherence.get("education_occupation_valid_rate_final"),
        "age_occupation_valid_rate_raw": coherence.get("age_occupation_valid_rate_raw"),
        "age_occupation_valid_rate_final": coherence.get("age_occupation_valid_rate_final"),
        "legacy_occupations_raw_count": len(occupation.get("legacy_occupations_raw", [])) if occupation else None,
        "legacy_occupations_final_count": len(occupation.get("legacy_occupations_final", [])) if occupation else None,
        "unicode_nfc_valid_raw": locale.get("unicode_nfc_valid_raw"),
        "unicode_nfc_valid_final": locale.get("unicode_nfc_valid_final"),
        "vocabulary_quality_gate_status": vocabulary_gates.get("status"),
    }


def flatten_run_metrics(
    benchmark_id: str,
    model: str,
    seed: int,
    result: dict[str, Any],
    train_size: int | None = None,
    holdout_size: int | None = None,
) -> list[dict[str, Any]]:
    """Achata saídas existentes de avaliação, validação e gates em linhas longas."""
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

    conditional_income = evaluation.get("conditional_income", {})
    conditional_summary = conditional_income.get("summary", {}) if isinstance(conditional_income, dict) else {}
    for metric_name in [
        "conditional_groups_compared",
        "max_conditional_income_wasserstein",
        "mean_conditional_income_wasserstein",
        "max_abs_p95_difference",
        "max_abs_p99_difference",
        "occupation_income_rank_correlation",
        "tail_events",
        "groups_with_excessive_tail",
    ]:
        add("conditional_income", metric_name, None, conditional_summary.get(metric_name))

    privacy = evaluation.get("privacy", {})
    duplicate_base = privacy.get("duplicate_base_rows") if isinstance(privacy.get("duplicate_base_rows"), dict) else {}
    exact_matches = privacy.get("exact_matches") if isinstance(privacy.get("exact_matches"), dict) else {}
    exact_train = exact_matches.get("train") if isinstance(exact_matches.get("train"), dict) else {}
    exact_holdout = exact_matches.get("holdout") if isinstance(exact_matches.get("holdout"), dict) else {}
    add("privacy", "duplicate_row_rate", None, privacy.get("duplicate_row_rate"))
    add("privacy", "duplicate_base_row_rate", None, duplicate_base.get("duplicate_row_rate"))
    add("privacy", "duplicate_base_duplicated_occurrences", None, duplicate_base.get("duplicated_occurrences"))
    add("privacy", "duplicate_base_duplicated_groups", None, duplicate_base.get("duplicated_groups"))
    add("privacy", "duplicate_base_largest_group", None, duplicate_base.get("largest_duplicate_group"))
    add("privacy", "unique_combination_rate", None, privacy.get("unique_combination_rate"))
    add("privacy", "unique_combinations", None, privacy.get("unique_combinations"))
    add("privacy", "exact_train_match_rate", None, privacy.get("exact_train_match_rate"))
    add("privacy", "exact_train_match_count", None, exact_train.get("exact_match_count"))
    add("privacy", "distinct_train_rows_matched", None, exact_train.get("distinct_reference_rows_matched"))
    add("privacy", "exact_holdout_match_rate", None, privacy.get("exact_holdout_match_rate"))
    add("privacy", "exact_holdout_match_count", None, exact_holdout.get("exact_match_count"))
    add("privacy", "distinct_holdout_rows_matched", None, exact_holdout.get("distinct_reference_rows_matched"))
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

    vocabulary = result.get("vocabulary_quality", {})
    if vocabulary:
        _add_vocabulary_long_rows(add, vocabulary)

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


def _add_vocabulary_long_rows(add, vocabulary: dict[str, Any]) -> None:
    original_add = add

    def add(
        metric_group: str,
        metric_name: str,
        column: str | None,
        value: Any,
        reference: Any = None,
        difference: Any = None,
        details: Any = None,
    ) -> None:
        safe_value = value
        safe_details = details
        if isinstance(value, bool):
            safe_value = int(value)
        elif isinstance(value, (str, dict, list, tuple)):
            safe_value = None
            safe_details = value if safe_details is None else safe_details
        original_add(metric_group, metric_name, column, safe_value, reference, difference, safe_details)

    occupation = vocabulary.get("occupation", {})
    coherence = vocabulary.get("coherence", {})
    rare = vocabulary.get("rare_occupations", {})
    income_by_occupation = vocabulary.get("income_by_occupation", {})
    comparisons = vocabulary.get("income_comparisons", {})
    diversity = vocabulary.get("diversity", {})
    locale = vocabulary.get("locale", {})
    gates = vocabulary.get("quality_gates", {})

    scalar_metrics = {
        "occupation_reference_coverage": occupation.get("occupation_reference_coverage"),
        "occupation_raw_coverage": occupation.get("occupation_raw_coverage"),
        "occupation_final_coverage": occupation.get("occupation_final_coverage"),
        "occupation_distribution_distance": occupation.get("occupation_distribution_distance"),
        "occupation_distribution_distance_raw": occupation.get("occupation_distribution_distance_raw"),
        "occupation_distribution_distance_final": occupation.get("occupation_distribution_distance_final"),
        "occupation_entropy_reference": occupation.get("occupation_entropy_reference"),
        "occupation_entropy_raw": occupation.get("occupation_entropy_raw"),
        "occupation_entropy_final": occupation.get("occupation_entropy_final"),
        "most_frequent_occupation_share_raw": occupation.get("most_frequent_occupation_share_raw"),
        "most_frequent_occupation_share_final": occupation.get("most_frequent_occupation_share_final"),
        "education_occupation_valid_rate_raw": coherence.get("education_occupation_valid_rate_raw"),
        "education_occupation_valid_rate_final": coherence.get("education_occupation_valid_rate_final"),
        "education_occupation_invalid_count_raw": coherence.get("education_occupation_invalid_count_raw"),
        "education_occupation_invalid_count_final": coherence.get("education_occupation_invalid_count_final"),
        "age_occupation_valid_rate_raw": coherence.get("age_occupation_valid_rate_raw"),
        "age_occupation_valid_rate_final": coherence.get("age_occupation_valid_rate_final"),
        "age_occupation_invalid_count_raw": coherence.get("age_occupation_invalid_count_raw"),
        "age_occupation_invalid_count_final": coherence.get("age_occupation_invalid_count_final"),
        "legacy_occupations_raw_count": len(occupation.get("legacy_occupations_raw", [])),
        "legacy_occupations_final_count": len(occupation.get("legacy_occupations_final", [])),
        "unicode_nfc_valid_raw": locale.get("unicode_nfc_valid_raw"),
        "unicode_nfc_valid_final": locale.get("unicode_nfc_valid_final"),
        "legacy_value_count_raw": locale.get("legacy_value_count_raw"),
        "legacy_value_count_final": locale.get("legacy_value_count_final"),
        "vocabulary_quality_gate_status": gates.get("status"),
    }
    for metric_name, value in scalar_metrics.items():
        add("vocabulary_summary", metric_name, None, value)
    add("vocabulary_summary", "missing_occupations_raw", None, len(occupation.get("missing_occupations_raw", [])), details=occupation.get("missing_occupations_raw", []))
    add("vocabulary_summary", "missing_occupations_final", None, len(occupation.get("missing_occupations_final", [])), details=occupation.get("missing_occupations_final", []))
    add("vocabulary_summary", "unexpected_occupations_raw", None, len(occupation.get("unexpected_occupations_raw", [])), details=occupation.get("unexpected_occupations_raw", []))
    add("vocabulary_summary", "unexpected_occupations_final", None, len(occupation.get("unexpected_occupations_final", [])), details=occupation.get("unexpected_occupations_final", []))
    add("vocabulary_summary", "legacy_occupations_raw", None, len(occupation.get("legacy_occupations_raw", [])), details=occupation.get("legacy_occupations_raw", []))
    add("vocabulary_summary", "legacy_occupations_final", None, len(occupation.get("legacy_occupations_final", [])), details=occupation.get("legacy_occupations_final", []))

    for metric_name in ["occupation_reference_count", "occupation_raw_count", "occupation_final_count"]:
        for occupation_name, count in occupation.get(metric_name, {}).items():
            add("vocabulary_occupation_distribution", metric_name, occupation_name, count)

    reference_counts = occupation.get("occupation_reference_count", {})
    raw_counts = occupation.get("occupation_raw_count", {})
    final_counts = occupation.get("occupation_final_count", {})
    for occupation_name in sorted(set(reference_counts) | set(raw_counts) | set(final_counts)):
        add(
            "vocabulary_occupation_coverage",
            "occupation_present_raw",
            occupation_name,
            bool(raw_counts.get(occupation_name, 0) > 0),
            reference=bool(reference_counts.get(occupation_name, 0) > 0),
        )
        add(
            "vocabulary_occupation_coverage",
            "occupation_present_final",
            occupation_name,
            bool(final_counts.get(occupation_name, 0) > 0),
            reference=bool(reference_counts.get(occupation_name, 0) > 0),
        )

    for occupation_name, row in rare.get("occupations", {}).items():
        for metric_name, value in row.items():
            add("vocabulary_rare_occupation", metric_name, occupation_name, value)

    for row in coherence.get("invalid_education_occupation_raw", []):
        column = f"{row.get('Escolaridade')} + {row.get('Ocupacao')}"
        add("vocabulary_invalid_education_occupation", "raw_invalid_count", column, row.get("count"), details=row)
    for row in coherence.get("invalid_education_occupation_final", []):
        column = f"{row.get('Escolaridade')} + {row.get('Ocupacao')}"
        add("vocabulary_invalid_education_occupation", "final_invalid_count", column, row.get("count"), details=row)
    for row in coherence.get("invalid_age_occupation_raw", []):
        column = f"{row.get('Idade')} + {row.get('Ocupacao')}"
        add("vocabulary_invalid_age_occupation", "raw_invalid_count", column, row.get("count"), details=row)
    for row in coherence.get("invalid_age_occupation_final", []):
        column = f"{row.get('Idade')} + {row.get('Ocupacao')}"
        add("vocabulary_invalid_age_occupation", "final_invalid_count", column, row.get("count"), details=row)

    for stage, summaries in income_by_occupation.items():
        for occupation_name, summary in summaries.items():
            for metric_name, value in summary.items():
                add("vocabulary_occupation_income", f"{stage}_{metric_name}", occupation_name, value)

    for comparison_name, stage_rows in comparisons.items():
        for stage, comparison in stage_rows.items():
            for metric_name, value in comparison.items():
                add("vocabulary_income_comparison", f"{stage}_{metric_name}", comparison_name, value)

    for stage, metrics in diversity.items():
        for metric_name, value in metrics.items():
            add("vocabulary_diversity", f"{stage}_{metric_name}", None, value)

    for stage, audit in vocabulary.get("gender_audit", {}).items():
        add("vocabulary_gender_audit", f"{stage}_income_by_gender", None, None, details=audit)
    add("vocabulary_gender_audit", "methodological_notice", None, None, details=vocabulary.get("methodological_notice"))

    for check in gates.get("blocking_checks", []):
        add("vocabulary_quality_gates", check.get("name"), None, check.get("passed"), details=check)
    for check in gates.get("diagnostic_checks", []):
        add("vocabulary_quality_gates", check.get("name"), None, check.get("value"), details=check)


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

    if config["benchmark"].get("type") == "vocabulary_quality":
        outputs.update(
            _write_vocabulary_quality_outputs(
                benchmark_dir=benchmark_dir,
                long_frame=long_frame,
                summary_frame=summary_frame,
                export_csv=export_csv,
                export_parquet=export_parquet,
                export_json=export_json,
            )
        )
    if config["benchmark"].get("type") == "income_realism":
        outputs.update(
            _write_income_realism_outputs(
                benchmark_dir=benchmark_dir,
                long_frame=long_frame,
                summary_frame=summary_frame,
                export_csv=export_csv,
                export_parquet=export_parquet,
                export_json=export_json,
            )
        )

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


def _write_vocabulary_quality_outputs(
    benchmark_dir: Path,
    long_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    export_csv: bool,
    export_parquet: bool,
    export_json: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if "metric_group" not in long_frame.columns:
        vocabulary_frame = pd.DataFrame(columns=long_frame.columns)
    else:
        vocabulary_frame = long_frame[long_frame["metric_group"].astype(str).str.startswith("vocabulary_")].copy()

    def filtered(group: str) -> pd.DataFrame:
        if vocabulary_frame.empty:
            return pd.DataFrame(columns=vocabulary_frame.columns)
        return vocabulary_frame[vocabulary_frame["metric_group"] == group].copy()

    datasets = {
        "vocabulary_v2_metrics": vocabulary_frame,
        "occupation_coverage": filtered("vocabulary_occupation_coverage"),
        "occupation_distribution": filtered("vocabulary_occupation_distribution"),
        "occupation_income_summary": filtered("vocabulary_occupation_income"),
        "invalid_education_occupation": filtered("vocabulary_invalid_education_occupation"),
        "invalid_age_occupation": filtered("vocabulary_invalid_age_occupation"),
        "rare_occupation_coverage": filtered("vocabulary_rare_occupation"),
    }
    if export_parquet:
        for name, frame in datasets.items():
            path = benchmark_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            outputs[f"{name}_parquet"] = path
    if export_csv:
        for name, frame in datasets.items():
            path = benchmark_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            outputs[f"{name}_csv"] = path
    if export_json:
        payload = {
            "summary": _json_safe_records(summary_frame),
            "interpretation": (
                "Benchmark de qualidade do vocabulário 2. Métricas raw usam a saída diagnóstica imediata "
                "do sintetizador salvo; métricas final usam o resultado pós-processado e validado pelo pipeline."
            ),
        }
        outputs["raw_vs_final_summary_json"] = write_json(payload, benchmark_dir / "raw_vs_final_summary.json")
    return outputs


def _write_income_realism_outputs(
    benchmark_dir: Path,
    long_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    export_csv: bool,
    export_parquet: bool,
    export_json: bool,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if "metric_group" not in long_frame.columns:
        income_frame = pd.DataFrame(columns=long_frame.columns)
    else:
        income_frame = long_frame[long_frame["metric_group"].astype(str).eq("conditional_income")].copy()
    summary_metrics = income_frame.pivot_table(
        index=["model", "seed", "train_size"],
        columns="metric_name",
        values="value",
        aggfunc="first",
    ).reset_index() if not income_frame.empty else pd.DataFrame()
    datasets = {
        "conditional_income_comparison": income_frame,
        "conditional_income_summary": summary_metrics,
        "conditional_income_tail_events": income_frame[
            income_frame.get("metric_name", pd.Series(dtype=str)).astype(str).eq("tail_events")
        ].copy() if not income_frame.empty else pd.DataFrame(columns=income_frame.columns),
    }
    if export_parquet:
        for name, frame in datasets.items():
            path = benchmark_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            outputs[f"{name}_parquet"] = path
    if export_csv:
        for name, frame in datasets.items():
            path = benchmark_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            outputs[f"{name}_csv"] = path
    if export_json:
        payload = {
            "summary": _json_safe_records(summary_frame),
            "conditional_income_metrics": _json_safe_records(summary_metrics),
            "interpretation": (
                "Benchmark exploratório de realismo condicional de renda. As linhas detalhadas por grupo "
                "ficam nos artefatos de cada execução do pipeline."
            ),
        }
        outputs["income_plausibility_summary_json"] = write_json(payload, benchmark_dir / "income_plausibility_summary.json")
    return outputs


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte registros tabulares para JSON sem valores não finitos."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if pd.isna(value):
            return None
        return value

    return [clean(record) for record in frame.to_dict(orient="records")]


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
