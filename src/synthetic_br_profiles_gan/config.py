"""Configuration loading and validation helpers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from synthetic_br_profiles_gan.exceptions import ConfigurationError


ConfigDict = dict[str, Any]


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Return a recursive merge of two configuration dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_config(path: str | Path) -> ConfigDict:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {config_path}")
    return loaded


def save_yaml_config(config: ConfigDict, path: str | Path) -> Path:
    """Persist a configuration dictionary as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return output_path


def require_keys(config: ConfigDict, keys: list[str], context: str) -> None:
    """Validate that all required keys exist in a dictionary."""
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(f"Missing required key(s) in {context}: {joined}")


def config_hash(config: ConfigDict) -> str:
    """Return a stable SHA256 hash for a configuration dictionary."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to an optional base directory."""
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def _reject_unknown(config: ConfigDict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown configuration key(s) in {context}: {', '.join(unknown)}")


def _require_positive_int(config: ConfigDict, key: str, context: str) -> None:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}.{key} must be an integer.") from exc
    if value <= 0:
        raise ConfigurationError(f"{context}.{key} must be greater than zero.")


def _require_non_negative_int(config: ConfigDict, key: str, context: str) -> None:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}.{key} must be an integer.") from exc
    if value < 0:
        raise ConfigurationError(f"{context}.{key} must be non-negative.")


def _require_bool(config: ConfigDict, key: str, context: str) -> None:
    if not isinstance(config.get(key), bool):
        raise ConfigurationError(f"{context}.{key} must be true or false.")


def validate_calibration_config(config: ConfigDict, context: str = "calibration") -> None:
    """Validate calibration configuration keys, types, and feasible ranges."""
    _reject_unknown(config, {"seed", "num_rows", "holdout_fraction", "income", "age", "region_weights"}, context)
    _require_positive_int(config, "num_rows", context)
    _require_non_negative_int(config, "seed", context)
    try:
        holdout_fraction = float(config["holdout_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}.holdout_fraction must be a float between 0 and 1.") from exc
    if not 0 < holdout_fraction < 1:
        raise ConfigurationError(f"{context}.holdout_fraction must be between 0 and 1.")

    income = config.get("income")
    if not isinstance(income, dict):
        raise ConfigurationError(f"{context}.income must be a mapping with min/max.")
    _reject_unknown(income, {"min", "max"}, f"{context}.income")
    income_min = float(income.get("min"))
    income_max = float(income.get("max"))
    if income_min < 0 or income_min >= income_max:
        raise ConfigurationError(f"{context}.income must satisfy 0 <= min < max.")

    age = config.get("age")
    if not isinstance(age, dict):
        raise ConfigurationError(f"{context}.age must be a mapping with min/max.")
    _reject_unknown(age, {"min", "max"}, f"{context}.age")
    age_min = int(age.get("min"))
    age_max = int(age.get("max"))
    if age_min < 0 or age_min >= age_max:
        raise ConfigurationError(f"{context}.age must satisfy 0 <= min < max.")

    weights = config.get("region_weights")
    if not isinstance(weights, dict):
        raise ConfigurationError(f"{context}.region_weights must be a mapping.")
    expected_regions = {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}
    _reject_unknown(weights, expected_regions, f"{context}.region_weights")
    missing = expected_regions - set(weights)
    if missing:
        raise ConfigurationError(f"Missing region weight(s) in {context}.region_weights: {', '.join(sorted(missing))}")
    total_weight = sum(float(value) for value in weights.values())
    if total_weight <= 0:
        raise ConfigurationError(f"{context}.region_weights must have positive total weight.")


def validate_generation_config(config: ConfigDict, context: str = "generation") -> None:
    """Validate generation configuration."""
    _reject_unknown(config, {"rows", "batch_size", "max_batches", "date_format"}, context)
    _require_positive_int(config, "rows", context)
    _require_positive_int(config, "batch_size", context)
    _require_positive_int(config, "max_batches", context)
    if not isinstance(config.get("date_format"), str) or not config["date_format"]:
        raise ConfigurationError(f"{context}.date_format must be a non-empty string.")


def validate_export_config(config: ConfigDict, context: str = "export") -> None:
    """Validate export configuration."""
    _reject_unknown(config, {"xlsx", "primary_format"}, context)
    _require_bool(config, "xlsx", context)
    if config.get("primary_format", "parquet") != "parquet":
        raise ConfigurationError(f"{context}.primary_format currently supports only 'parquet'.")


def validate_quality_gate_config(config: ConfigDict, context: str = "quality_gates") -> None:
    """Validate quality gate configuration shape."""
    allowed = {
        "assessment_mode",
        "min_evaluation_rows",
        "invalid_rows_max",
        "duplicated_identifier_max",
        "null_required_fields_max",
        "exact_train_match_rate_max",
        "total_variation_distance_max",
        "correlation_difference_max",
    }
    _reject_unknown(config, allowed, context)
    if config.get("assessment_mode", "experimental") not in {"smoke", "experimental", "approval"}:
        raise ConfigurationError(f"{context}.assessment_mode must be smoke, experimental, or approval.")
    if "min_evaluation_rows" in config:
        _require_positive_int(config, "min_evaluation_rows", context)
    for key, value in config.items():
        if key in {"assessment_mode", "min_evaluation_rows"}:
            continue
        if isinstance(value, dict):
            _reject_unknown(value, {"value", "mandatory", "description"}, f"{context}.{key}")
            if "value" not in value:
                raise ConfigurationError(f"{context}.{key}.value is required.")
            if "mandatory" in value and not isinstance(value["mandatory"], bool):
                raise ConfigurationError(f"{context}.{key}.mandatory must be true or false.")
        elif not isinstance(value, (int, float)):
            raise ConfigurationError(f"{context}.{key} must be a number or a value/mandatory mapping.")


def validate_model_config(model_name: str, config: ConfigDict) -> None:
    """Validate synthesizer-specific configuration."""
    normalized = model_name.lower().replace("-", "_")
    if normalized == "programmatic":
        validate_calibration_config(config, "models.programmatic")
        return
    if normalized in {"simple_gan", "simple_tabular_gan", "dense_tabular_gan"}:
        allowed = {"seed", "latent_dim", "epochs", "batch_size", "learning_rate", "beta_1", "verbose_every", "metrics_every"}
        _reject_unknown(config, allowed, "models.simple_gan")
        for key in ["latent_dim", "epochs", "batch_size"]:
            _require_positive_int(config, key, "models.simple_gan")
        for key in ["verbose_every", "metrics_every", "seed"]:
            _require_non_negative_int(config, key, "models.simple_gan")
        if float(config["learning_rate"]) <= 0:
            raise ConfigurationError("models.simple_gan.learning_rate must be greater than zero.")
        beta_1 = float(config["beta_1"])
        if not 0 <= beta_1 < 1:
            raise ConfigurationError("models.simple_gan.beta_1 must be in [0, 1).")
        return
    if normalized in {"ctgan", "ctgan_synthesizer"}:
        allowed = {"seed", "epochs", "batch_size", "verbose", "enable_gpu", "cuda", "library"}
        _reject_unknown(config, allowed, "models.ctgan")
        for key in ["epochs", "batch_size"]:
            _require_positive_int(config, key, "models.ctgan")
        _require_non_negative_int(config, "seed", "models.ctgan")
        _require_bool(config, "verbose", "models.ctgan")
        if "enable_gpu" in config:
            _require_bool(config, "enable_gpu", "models.ctgan")
        if config.get("cuda") is not None and not isinstance(config.get("cuda"), bool):
            raise ConfigurationError("models.ctgan.cuda must be true, false, or null.")
        if "library" in config and not isinstance(config["library"], dict):
            raise ConfigurationError("models.ctgan.library must be a mapping when provided.")
        return
    raise ConfigurationError(f"Unknown model configuration: {model_name}")


def validate_pipeline_config(config: ConfigDict) -> None:
    """Validate the resolved pipeline configuration before execution."""
    _reject_unknown(
        config,
        {
            "seed",
            "artifacts_root",
            "reference_date",
            "model",
            "calibration",
            "models",
            "generation",
            "evaluation",
            "quality_gates",
            "export",
        },
        "pipeline",
    )
    _require_non_negative_int(config, "seed", "pipeline")
    if config.get("model") not in {"programmatic", "simple_gan", "ctgan"}:
        raise ConfigurationError("pipeline.model must be one of: programmatic, simple_gan, ctgan.")
    if not isinstance(config.get("artifacts_root"), str) or not config["artifacts_root"]:
        raise ConfigurationError("pipeline.artifacts_root must be a non-empty path string.")
    if not isinstance(config.get("reference_date"), str) or not config["reference_date"]:
        raise ConfigurationError("pipeline.reference_date must be a YYYY-MM-DD string.")
    validate_calibration_config(config["calibration"], "calibration")
    validate_generation_config(config["generation"], "generation")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ConfigurationError("pipeline.evaluation must be a mapping when provided.")
    _reject_unknown(evaluation, {"privacy"}, "evaluation")
    privacy = evaluation.get("privacy", {})
    if not isinstance(privacy, dict):
        raise ConfigurationError("evaluation.privacy must be a mapping when provided.")
    _reject_unknown(privacy, {"max_nearest_neighbor_rows", "exclude_columns"}, "evaluation.privacy")
    if "max_nearest_neighbor_rows" in privacy:
        _require_positive_int(privacy, "max_nearest_neighbor_rows", "evaluation.privacy")
    validate_quality_gate_config(config["quality_gates"], "quality_gates")
    validate_export_config(config["export"], "export")
    models = config.get("models")
    if not isinstance(models, dict):
        raise ConfigurationError("pipeline.models must be a mapping.")
    selected = str(config["model"])
    if selected not in models:
        raise ConfigurationError(f"pipeline.models must contain configuration for selected model '{selected}'.")
