"""Configuração da interface Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.column_catalog import available_presets
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge, load_yaml_config
from synthetic_br_profiles_gan.exceptions import ConfigurationError


SUPPORTED_UI_FORMATS = ("csv", "json", "parquet")
SUPPORTED_UI_MODELS = ("programmatic", "ctgan", "simple_gan")


DEFAULT_UI_CONFIG: ConfigDict = {
    "application": {
        "title": "Gerador de Perfis Sintéticos Brasileiros",
        "preview_rows": 20,
        "models_root": "artifacts/models",
        "sessions_root": "artifacts/ui_sessions",
    },
    "generation": {
        "default_rows": 1000,
        "min_rows": 1,
        "limits": {
            "programmatic": 100000,
            "ctgan": 50000,
            "simple_gan": 20000,
        },
    },
    "defaults": {
        "model": "programmatic",
        "preset": "completo",
        "format": "csv",
        "seed": 41,
    },
}


@dataclass(frozen=True)
class UIConfig:
    """Configuração resolvida da interface de geração."""

    title: str
    preview_rows: int
    models_root: Path
    sessions_root: Path
    default_rows: int
    min_rows: int
    limits: dict[str, int]
    default_model: str
    default_preset: str
    default_format: str
    default_seed: int
    raw: dict[str, Any]


def load_ui_config(path: str | Path = "configs/ui.yaml") -> UIConfig:
    """Carrega e valida a configuração da interface."""
    loaded = load_yaml_config(path)
    effective = deep_merge(DEFAULT_UI_CONFIG, loaded)
    validate_ui_config(effective)
    application = effective["application"]
    generation = effective["generation"]
    defaults = effective["defaults"]
    return UIConfig(
        title=str(application["title"]),
        preview_rows=int(application["preview_rows"]),
        models_root=Path(application["models_root"]),
        sessions_root=Path(application["sessions_root"]),
        default_rows=int(generation["default_rows"]),
        min_rows=int(generation["min_rows"]),
        limits={model: int(limit) for model, limit in generation["limits"].items()},
        default_model=str(defaults["model"]),
        default_preset=str(defaults["preset"]),
        default_format=str(defaults["format"]),
        default_seed=int(defaults["seed"]),
        raw=effective,
    )


def validate_ui_config(config: ConfigDict) -> None:
    """Valida chaves, tipos e limites operacionais da interface."""
    _reject_unknown(config, {"application", "generation", "defaults"}, "ui")
    application = _mapping(config.get("application"), "application")
    generation = _mapping(config.get("generation"), "generation")
    defaults = _mapping(config.get("defaults"), "defaults")

    _reject_unknown(application, {"title", "preview_rows", "models_root", "sessions_root"}, "application")
    if not isinstance(application.get("title"), str) or not application["title"]:
        raise ConfigurationError("application.title deve ser uma string não vazia.")
    _positive_int(application, "preview_rows", "application")
    for key in ("models_root", "sessions_root"):
        if not isinstance(application.get(key), str) or not application[key]:
            raise ConfigurationError(f"application.{key} deve ser um caminho não vazio.")

    _reject_unknown(generation, {"default_rows", "min_rows", "limits"}, "generation")
    _positive_int(generation, "default_rows", "generation")
    _positive_int(generation, "min_rows", "generation")
    limits = _mapping(generation.get("limits"), "generation.limits")
    _reject_unknown(limits, set(SUPPORTED_UI_MODELS), "generation.limits")
    missing = set(SUPPORTED_UI_MODELS) - set(limits)
    if missing:
        raise ConfigurationError(f"generation.limits deve conter: {', '.join(sorted(missing))}.")
    for model in SUPPORTED_UI_MODELS:
        _positive_int(limits, model, "generation.limits")
    if int(generation["default_rows"]) < int(generation["min_rows"]):
        raise ConfigurationError("generation.default_rows deve ser maior ou igual a generation.min_rows.")

    _reject_unknown(defaults, {"model", "preset", "format", "seed"}, "defaults")
    if defaults.get("model") not in SUPPORTED_UI_MODELS:
        raise ConfigurationError("defaults.model deve ser programmatic, ctgan ou simple_gan.")
    if defaults.get("preset") not in available_presets():
        raise ConfigurationError("defaults.preset deve ser um preset conhecido.")
    if defaults.get("format") not in SUPPORTED_UI_FORMATS:
        raise ConfigurationError("defaults.format deve ser csv, json ou parquet.")
    _non_negative_int(defaults, "seed", "defaults")
    if int(generation["default_rows"]) > int(limits[str(defaults["model"])]):
        raise ConfigurationError("generation.default_rows excede o limite operacional do modelo padrão.")


def _mapping(value: Any, context: str) -> ConfigDict:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} deve ser um mapeamento.")
    return value


def _reject_unknown(config: ConfigDict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ConfigurationError(f"Chaves desconhecidas em {context}: {', '.join(unknown)}.")


def _positive_int(config: ConfigDict, key: str, context: str) -> None:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}.{key} deve ser um inteiro.") from exc
    if value <= 0:
        raise ConfigurationError(f"{context}.{key} deve ser maior que zero.")


def _non_negative_int(config: ConfigDict, key: str, context: str) -> None:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{context}.{key} deve ser um inteiro.") from exc
    if value < 0:
        raise ConfigurationError(f"{context}.{key} deve ser maior ou igual a zero.")
