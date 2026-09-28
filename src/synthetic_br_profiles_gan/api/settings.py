"""Configuração da API HTTP sem dependência da camada de UI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.config import load_yaml_config
from synthetic_br_profiles_gan.exceptions import ConfigurationError


SUPPORTED_MODELS = ("programmatic", "ctgan", "simple_gan")
SUPPORTED_OUTPUT_FORMATS = ("csv", "json", "parquet")
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True)
class ApiSettings:
    """Parâmetros operacionais da API usada pela interface React."""

    title: str = "Plataforma de Dados Sintéticos Brasileiros"
    preview_rows: int = 20
    models_root: Path = Path("artifacts/models")
    web_sessions_root: Path = Path("artifacts/web_sessions")
    artifacts_root: Path = Path("artifacts")
    default_rows: int = 1000
    min_rows: int = 1
    row_limits: dict[str, int] = field(
        default_factory=lambda: {"programmatic": 100000, "ctgan": 50000, "simple_gan": 20000}
    )
    default_model: str = "programmatic"
    default_preset: str = "completo"
    default_format: str = "csv"
    default_seed: int = 41
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    max_workers: int = 1


def load_api_settings(path: str | Path = "configs/ui.yaml") -> ApiSettings:
    """Carrega a configuração compartilhada com a UI sem importar módulos Streamlit."""
    config_path = Path(path)
    if config_path.exists():
        config = load_yaml_config(config_path)
    else:
        config = {}
    application = _mapping(config.get("application", {}), "application")
    generation = _mapping(config.get("generation", {}), "generation")
    defaults = _mapping(config.get("defaults", {}), "defaults")
    limits = _mapping(generation.get("limits", {}), "generation.limits")

    cors_origins = _cors_origins_from_environment()
    settings = ApiSettings(
        title=str(application.get("title", ApiSettings.title)),
        preview_rows=_positive_int(application.get("preview_rows", 20), "application.preview_rows"),
        models_root=Path(str(application.get("models_root", "artifacts/models"))),
        web_sessions_root=Path(str(application.get("web_sessions_root", application.get("sessions_root", "artifacts/web_sessions")))),
        artifacts_root=Path(str(application.get("artifacts_root", "artifacts"))),
        default_rows=_positive_int(generation.get("default_rows", 1000), "generation.default_rows"),
        min_rows=_positive_int(generation.get("min_rows", 1), "generation.min_rows"),
        row_limits={
            model: _positive_int(limits.get(model, ApiSettings().row_limits[model]), f"generation.limits.{model}")
            for model in SUPPORTED_MODELS
        },
        default_model=str(defaults.get("model", "programmatic")).lower().replace("-", "_"),
        default_preset=str(defaults.get("preset", "completo")),
        default_format=str(defaults.get("format", "csv")).lower(),
        default_seed=_non_negative_int(defaults.get("seed", 41), "defaults.seed"),
        cors_origins=cors_origins,
    )
    _validate_settings(settings)
    return settings


def _cors_origins_from_environment() -> tuple[str, ...]:
    raw_value = os.getenv("SYNTHETIC_BR_PROFILES_GAN_CORS_ORIGINS", "")
    if not raw_value.strip():
        return DEFAULT_CORS_ORIGINS
    origins = tuple(origin.strip() for origin in raw_value.split(",") if origin.strip())
    if not origins or "*" in origins:
        raise ConfigurationError("CORS origins must be explicit; '*' is not allowed for the React API.")
    return origins


def _validate_settings(settings: ApiSettings) -> None:
    if settings.default_model not in SUPPORTED_MODELS:
        raise ConfigurationError(f"Modelo padrão desconhecido na API: {settings.default_model}")
    if settings.default_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ConfigurationError(f"Formato padrão desconhecido na API: {settings.default_format}")
    for model in SUPPORTED_MODELS:
        if settings.row_limits[model] < settings.min_rows:
            raise ConfigurationError(f"generation.limits.{model} deve ser maior ou igual a generation.min_rows.")
    if not settings.cors_origins or "*" in settings.cors_origins:
        raise ConfigurationError("A API exige origens CORS explícitas.")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} deve ser um mapeamento.")
    return value


def _positive_int(value: Any, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context} deve ser um inteiro positivo.") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{context} deve ser maior que zero.")
    return parsed


def _non_negative_int(value: Any, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context} deve ser um inteiro não negativo.") from exc
    if parsed < 0:
        raise ConfigurationError(f"{context} deve ser não negativo.")
    return parsed
