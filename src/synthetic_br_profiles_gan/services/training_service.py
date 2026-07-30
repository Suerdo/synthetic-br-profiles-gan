"""Serviço reutilizável de treinamento e serialização de sintetizadores."""

from __future__ import annotations

import importlib.metadata
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.calibration import (
    DEFAULT_CALIBRATION_CONFIG,
    generate_calibration_dataset,
    split_train_holdout,
)
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge, save_yaml_config
from synthetic_br_profiles_gan.domain.geography import (
    GEOGRAPHY_CATALOG_VERSION,
    GEOGRAPHY_MODEL_VERSION,
    LEGACY_GEOGRAPHY_MODEL_VERSION,
    geography_catalog_checksum,
)
from synthetic_br_profiles_gan.exceptions import ConfigurationError
from synthetic_br_profiles_gan.manifest import environment_info, get_git_commit, write_json
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.localization import (
    CATEGORICAL_VOCABULARY_VERSION,
    DATA_LOCALE,
    INCOME_MODEL_VERSION,
    UNICODE_NORMALIZATION,
)
from synthetic_br_profiles_gan.pipeline import DEFAULT_PIPELINE_CONFIG, train_synthesizer
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed


SUPPORTED_TRAINING_MODELS = {"programmatic", "simple_gan", "ctgan"}


@dataclass(frozen=True)
class TrainingRequest:
    """Solicitação de treinamento independente da CLI."""

    model: str
    output_path: Path
    config: dict[str, Any]
    seed: int = 41
    train_rows: int = 1000
    holdout_fraction: float = 0.20
    overwrite: bool = False
    calibration_path: Path | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Resultado estruturado de um treinamento ou artefato programático."""

    model: str
    output_path: Path
    train_rows: int
    holdout_rows: int
    duration_seconds: float
    model_size_bytes: int
    manifest_path: Path


def run_training(request: TrainingRequest) -> TrainingResult:
    """Executa preparação de dados, ajuste do sintetizador e gravação de manifesto."""
    started = datetime.now(timezone.utc)
    perf_start = time.perf_counter()
    model = _normalize_model_name(request.model)
    _validate_training_request(request, model)
    output_path = Path(request.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = default_metadata()
    set_global_seed(
        int(request.seed),
        seed_tensorflow=model == "simple_gan",
        seed_torch=model == "ctgan",
    )

    calibration_started = time.perf_counter()
    if request.calibration_path is not None:
        train = _read_table(request.calibration_path)
        holdout = pd.DataFrame(columns=train.columns)
        calibration_rows = int(len(train))
        holdout_rows = 0
        calibration_seconds = float(time.perf_counter() - calibration_started)
    else:
        holdout_rows = holdout_rows_for_train_size(int(request.train_rows), float(request.holdout_fraction))
        calibration_rows = int(request.train_rows) + holdout_rows
        calibration_config = _training_calibration_config(request, calibration_rows)
        calibration = generate_calibration_dataset(config=calibration_config)
        calibration_seconds = float(time.perf_counter() - calibration_started)
        train, holdout = split_train_holdout(
            calibration,
            holdout_fraction=float(request.holdout_fraction),
            seed=int(request.seed),
            train_rows=int(request.train_rows),
            holdout_rows=holdout_rows,
        )

    training_started = time.perf_counter()
    training_required = model != "programmatic"
    model_config = _training_model_config(model, request)
    train_synthesizer(model, train, metadata, config=model_config, output_dir=output_path)
    training_seconds = float(time.perf_counter() - training_started)

    serialization_started = time.perf_counter()
    metadata.save(output_path / "metadata.json")
    config_path = save_yaml_config(_resolved_training_config(request, model, calibration_rows), output_path / "training_config.yaml")
    serialization_seconds = float(time.perf_counter() - serialization_started)
    model_size_bytes = _path_size_bytes(output_path)
    ended = datetime.now(timezone.utc)
    manifest = build_training_manifest(
        model=model,
        seed=int(request.seed),
        training_required=training_required,
        train_rows=int(len(train)),
        holdout_rows=int(len(holdout)),
        calibration_rows=calibration_rows,
        metadata=metadata,
        config=_resolved_training_config(request, model, calibration_rows),
        started_at=started,
        ended_at=ended,
        timings={
            "calibration_seconds": calibration_seconds,
            "training_seconds": training_seconds if training_required else 0.0,
            "serialization_seconds": serialization_seconds,
            "total_seconds": float(time.perf_counter() - perf_start),
        },
        model_size_bytes=model_size_bytes,
        config_path=config_path,
    )
    manifest_path = write_json(manifest, output_path / "training_manifest.json")
    return TrainingResult(
        model=model,
        output_path=output_path,
        train_rows=int(len(train)),
        holdout_rows=int(len(holdout)),
        duration_seconds=float((ended - started).total_seconds()),
        model_size_bytes=model_size_bytes,
        manifest_path=manifest_path,
    )


def build_training_manifest(
    model: str,
    seed: int,
    training_required: bool,
    train_rows: int,
    holdout_rows: int,
    calibration_rows: int,
    metadata: DatasetMetadata,
    config: dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
    timings: dict[str, float],
    model_size_bytes: int,
    config_path: Path,
) -> dict[str, Any]:
    """Cria o manifesto de treinamento de um sintetizador salvo."""
    env = environment_info()
    libraries = env.get("library_versions", {})
    return {
        "schema_version": 1,
        "artifact_type": "trained_synthesizer",
        "model": model,
        "created_at_utc": started_at.astimezone(timezone.utc).isoformat(),
        "ended_at_utc": ended_at.astimezone(timezone.utc).isoformat(),
        "project_version": _project_version(),
        "data_locale": DATA_LOCALE,
        "unicode_normalization": UNICODE_NORMALIZATION,
        "categorical_vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "income_model_version": _income_model_version_from_config(config),
        "geography_model_version": _geography_model_version_from_config(model, config),
        "geography_generation_strategy": "direct_rules" if model == "programmatic" else "model_columns",
        "geography_catalog_version": (
            GEOGRAPHY_CATALOG_VERSION if _geography_model_version_from_config(model, config) == GEOGRAPHY_MODEL_VERSION else None
        ),
        "geography_catalog_checksum": (
            geography_catalog_checksum() if _geography_model_version_from_config(model, config) == GEOGRAPHY_MODEL_VERSION else None
        ),
        "seed": int(seed),
        "training_required": bool(training_required),
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "calibration_rows": int(calibration_rows),
        "model_columns": list(metadata.model_columns),
        "final_columns": list(metadata.final_columns),
        "config": config,
        "training_config": str(config_path),
        "environment": {
            "python": env.get("python_version"),
            "platform": env.get("platform"),
            "cpu_count": os.cpu_count(),
            "gpu_available": _gpu_available(env),
            "libraries": libraries,
        },
        "timings": timings,
        "model_size_bytes": int(model_size_bytes),
        "git_commit": get_git_commit(Path.cwd()),
    }


def holdout_rows_for_train_size(train_rows: int, holdout_fraction: float) -> int:
    """Calcula o holdout exato implícito por um tamanho de treino."""
    if int(train_rows) <= 0:
        raise ConfigurationError("train_rows must be greater than zero.")
    if not 0 < float(holdout_fraction) < 1:
        raise ConfigurationError("holdout_fraction must be between 0 and 1.")
    value = int(train_rows) * float(holdout_fraction) / (1.0 - float(holdout_fraction))
    rounded = int(round(value))
    if abs(value - rounded) > 1e-9:
        raise ConfigurationError("train_rows and holdout_fraction must produce an exact integer holdout size.")
    return rounded


def _validate_training_request(request: TrainingRequest, model: str) -> None:
    if model not in SUPPORTED_TRAINING_MODELS:
        raise ConfigurationError(f"Unknown training model: {request.model}")
    if int(request.seed) < 0:
        raise ConfigurationError("seed must be a non-negative integer.")
    if int(request.train_rows) <= 0:
        raise ConfigurationError("train_rows must be greater than zero.")
    if not 0 < float(request.holdout_fraction) < 1:
        raise ConfigurationError("holdout_fraction must be between 0 and 1.")
    output_path = Path(request.output_path)
    if output_path.exists() and any(output_path.iterdir()) and not request.overwrite:
        raise ConfigurationError(f"Output directory already exists and is not empty: {output_path}")
    if request.calibration_path is not None and not Path(request.calibration_path).exists():
        raise ConfigurationError(f"Calibration file not found: {request.calibration_path}")


def _normalize_model_name(model: str) -> str:
    normalized = model.lower().replace("-", "_")
    aliases = {
        "simple_tabular_gan": "simple_gan",
        "dense_tabular_gan": "simple_gan",
        "ctgan_synthesizer": "ctgan",
        "programmatic_synthesizer": "programmatic",
    }
    return aliases.get(normalized, normalized)


def _training_calibration_config(request: TrainingRequest, calibration_rows: int) -> ConfigDict:
    calibration = dict(request.config.get("calibration", {}))
    calibration.update({"seed": int(request.seed), "num_rows": int(calibration_rows), "holdout_fraction": float(request.holdout_fraction)})
    return deep_merge(DEFAULT_CALIBRATION_CONFIG, calibration)


def _training_model_config(model: str, request: TrainingRequest) -> ConfigDict:
    defaults = DEFAULT_PIPELINE_CONFIG.get("models", {}).get(model, {})
    configured = request.config.get("models", {}).get(model, {})
    seed = int(request.seed) + 100_003 if model == "programmatic" else int(request.seed)
    if model == "programmatic":
        return deep_merge(_training_calibration_config(request, int(request.train_rows)), {**configured, "seed": seed})
    return deep_merge(defaults, {**configured, "seed": seed})


def _resolved_training_config(request: TrainingRequest, model: str, calibration_rows: int) -> dict[str, Any]:
    return {
        "model": model,
        "seed": int(request.seed),
        "train_rows": int(request.train_rows),
        "holdout_fraction": float(request.holdout_fraction),
        "calibration_rows": int(calibration_rows),
        "calibration": _training_calibration_config(request, calibration_rows),
        "models": {model: _training_model_config(model, request)},
    }


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ConfigurationError(f"Unsupported calibration format: {path}")


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _project_version() -> str | None:
    try:
        return importlib.metadata.version("synthetic-br-profiles-gan")
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_available(env: dict[str, Any]) -> bool:
    gpu = env.get("gpu", {})
    tensorflow_gpu = gpu.get("tensorflow")
    torch_cuda = gpu.get("torch_cuda_available")
    return bool((isinstance(tensorflow_gpu, list) and tensorflow_gpu) or torch_cuda is True)


def _income_model_version_from_config(config: dict[str, Any]) -> int:
    try:
        calibration = config.get("calibration", {})
        return int(calibration.get("income_model_version", INCOME_MODEL_VERSION))
    except (AttributeError, TypeError, ValueError):
        return INCOME_MODEL_VERSION


def _geography_model_version_from_config(model: str, config: dict[str, Any]) -> int:
    if model != "ctgan":
        return LEGACY_GEOGRAPHY_MODEL_VERSION
    try:
        model_config = config.get("models", {}).get("ctgan", {})
        return int(model_config.get("geography_model_version", LEGACY_GEOGRAPHY_MODEL_VERSION))
    except (AttributeError, TypeError, ValueError):
        return LEGACY_GEOGRAPHY_MODEL_VERSION
