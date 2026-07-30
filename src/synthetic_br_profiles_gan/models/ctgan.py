"""Wrapper do sintetizador CTGAN usando o pacote standalone ctgan."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.calibration import coerce_model_dtypes
from synthetic_br_profiles_gan.config import ConfigDict
from synthetic_br_profiles_gan.domain.geography import (
    GEO_KEY_COLUMN,
    GEOGRAPHY_CATALOG_VERSION,
    GEOGRAPHY_MODEL_VERSION,
    LEGACY_GEOGRAPHY_MODEL_VERSION,
    build_geography_catalog,
    decode_geography_frame,
    encode_geography_frame,
    geography_catalog_checksum,
    geography_catalog_records,
    geography_v2_metadata,
    validate_geography_mapping,
)
from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable, ModelSerializationError, SyntheticModelError
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed


DEFAULT_CTGAN_CONFIG: ConfigDict = {
    "seed": 41,
    "epochs": 300,
    "batch_size": 500,
    "verbose": False,
    "enable_gpu": False,
    "cuda": None,
    "embedding_dim": None,
    "generator_dim": None,
    "discriminator_dim": None,
    "generator_lr": None,
    "discriminator_lr": None,
    "generator_decay": None,
    "discriminator_decay": None,
    "discriminator_steps": None,
    "log_frequency": None,
    "pac": None,
    "geography_model_version": LEGACY_GEOGRAPHY_MODEL_VERSION,
}

CTGAN_OPTIONAL_KWARGS = {
    "embedding_dim",
    "generator_dim",
    "discriminator_dim",
    "generator_lr",
    "discriminator_lr",
    "generator_decay",
    "discriminator_decay",
    "discriminator_steps",
    "log_frequency",
    "pac",
}


class CTGANSynthesizer:
    """Implementação real de CTGAN baseada no pacote standalone ``ctgan``."""

    model_name = "ctgan"

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = {**DEFAULT_CTGAN_CONFIG, **(config or {})}
        self.metadata: DatasetMetadata = default_metadata()
        self.training_metadata: DatasetMetadata = self.metadata
        self.model: Any | None = None
        self.discrete_columns: list[str] = []
        self.library_version: str | None = None
        self.geography_model_version = int(self.config.get("geography_model_version", LEGACY_GEOGRAPHY_MODEL_VERSION))
        self.geography_catalog_version: int | None = None
        self.geography_catalog_checksum: str | None = None

    @staticmethod
    def _ctgan_class():
        try:
            from ctgan import CTGAN
        except ImportError as exc:
            raise ModelBackendUnavailable(
                "The standalone ctgan package is required for CTGANSynthesizer. "
                "Install with: pip install -e \".[ctgan]\""
            ) from exc
        return CTGAN

    def fit(self, data: pd.DataFrame, metadata: DatasetMetadata) -> None:
        """Ajusta a CTGAN declarando explicitamente colunas categóricas e discretas."""
        set_global_seed(int(self.config.get("seed", 41)), seed_tensorflow=False, seed_torch=True)
        CTGAN = self._ctgan_class()
        self.metadata = metadata
        self.geography_model_version = int(self.config.get("geography_model_version", LEGACY_GEOGRAPHY_MODEL_VERSION))
        if self.geography_model_version == GEOGRAPHY_MODEL_VERSION:
            mapping = validate_geography_mapping()
            if not mapping["is_valid"]:
                raise SyntheticModelError("Invalid geography catalog for CTGAN geography_model_version=2.")
            self.training_metadata = geography_v2_metadata(metadata)
            self.geography_catalog_version = GEOGRAPHY_CATALOG_VERSION
            self.geography_catalog_checksum = geography_catalog_checksum()
            train = coerce_model_dtypes(encode_geography_frame(data[metadata.model_columns]), self.training_metadata)
        elif self.geography_model_version == LEGACY_GEOGRAPHY_MODEL_VERSION:
            self.training_metadata = metadata
            train = coerce_model_dtypes(data[metadata.model_columns], metadata)
        else:
            raise SyntheticModelError(f"Unsupported CTGAN geography_model_version: {self.geography_model_version}")
        self.discrete_columns = self.training_metadata.categorical_columns(include_discrete_numeric=True)
        self.library_version = importlib.metadata.version("ctgan")
        model_kwargs = _ctgan_constructor_kwargs(CTGAN, self.config)
        self.model = CTGAN(**model_kwargs)
        self.model.fit(train, self.discrete_columns)

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Amostra linhas canônicas do modelo e restaura os dtypes configurados."""
        if self.model is None:
            raise SyntheticModelError("CTGANSynthesizer must be fitted or loaded before sampling.")
        sampled = self.model.sample(int(num_rows))
        if self.geography_model_version == GEOGRAPHY_MODEL_VERSION:
            sampled = decode_geography_frame(sampled)
        else:
            sampled = sampled[self.metadata.model_columns]
        return coerce_model_dtypes(sampled, self.metadata)

    def save(self, output_path: Path) -> None:
        """Salva o modelo CTGAN, os metadados e a configuração de ajuste."""
        if self.model is None:
            raise SyntheticModelError("Cannot save an unfitted CTGANSynthesizer.")
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "model.pkl").open("wb") as file:
            pickle.dump(self.model, file)
        self.metadata.save(output_path / "metadata.json")
        if self.geography_model_version == GEOGRAPHY_MODEL_VERSION:
            self.training_metadata.save(output_path / "metadata_ctgan_internal.json")
            with (output_path / "geography_catalog.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "geography_model_version": GEOGRAPHY_MODEL_VERSION,
                        "geography_catalog_version": GEOGRAPHY_CATALOG_VERSION,
                        "checksum": geography_catalog_checksum(),
                        "entries": geography_catalog_records(),
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
        payload = {
            "config": self.config,
            "discrete_columns": self.discrete_columns,
            "library": "ctgan",
            "library_version": self.library_version or importlib.metadata.version("ctgan"),
            "geography_model_version": int(self.geography_model_version),
            "geography_catalog_version": self.geography_catalog_version,
            "geography_catalog_checksum": self.geography_catalog_checksum,
            "external_model_columns": list(self.metadata.model_columns),
            "training_model_columns": list(self.training_metadata.model_columns),
            "geo_key_column": GEO_KEY_COLUMN if self.geography_model_version == GEOGRAPHY_MODEL_VERSION else None,
        }
        with (output_path / "metadata_ctgan.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "CTGANSynthesizer":
        """Carrega um sintetizador CTGAN salvo."""
        cls._ctgan_class()
        try:
            with (input_path / "metadata_ctgan.json").open(encoding="utf-8") as file:
                payload = json.load(file)
            instance = cls(config=payload.get("config", {}))
            instance.discrete_columns = list(payload.get("discrete_columns", []))
            instance.library_version = payload.get("library_version")
            instance.metadata = DatasetMetadata.load(input_path / "metadata.json")
            instance.geography_model_version = int(payload.get("geography_model_version", LEGACY_GEOGRAPHY_MODEL_VERSION))
            instance.geography_catalog_version = payload.get("geography_catalog_version")
            instance.geography_catalog_checksum = payload.get("geography_catalog_checksum")
            if instance.geography_model_version == GEOGRAPHY_MODEL_VERSION:
                _validate_saved_geography_catalog(input_path, payload)
                internal_path = input_path / "metadata_ctgan_internal.json"
                instance.training_metadata = DatasetMetadata.load(internal_path) if internal_path.exists() else geography_v2_metadata(instance.metadata)
            else:
                instance.training_metadata = instance.metadata
            with (input_path / "model.pkl").open("rb") as file:
                instance.model = pickle.load(file)
            return instance
        except Exception as exc:
            raise ModelSerializationError(f"Could not load CTGANSynthesizer from {input_path}: {exc}") from exc


def _ctgan_constructor_kwargs(ctgan_class: Any, config: ConfigDict) -> dict[str, Any]:
    """Monta kwargs compatíveis com a assinatura da versão instalada da CTGAN."""
    signature = inspect.signature(ctgan_class)
    supported = set(signature.parameters)
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    batch_size = int(config["batch_size"])
    pac = config.get("pac")
    if pac is not None and (accepts_var_kwargs or "pac" in supported):
        pac_int = int(pac)
        if pac_int <= 0:
            raise SyntheticModelError("ctgan.pac must be greater than zero.")
        if batch_size % pac_int != 0:
            raise SyntheticModelError("ctgan.batch_size must be divisible by ctgan.pac.")
    kwargs = {
        "epochs": int(config["epochs"]),
        "batch_size": batch_size,
        "verbose": bool(config["verbose"]),
        "enable_gpu": bool(config.get("enable_gpu", config.get("cuda", False))),
    }
    for key in CTGAN_OPTIONAL_KWARGS:
        if (accepts_var_kwargs or key in supported) and config.get(key) is not None:
            kwargs[key] = config[key]
    if accepts_var_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in supported}


def _validate_saved_geography_catalog(input_path: Path, payload: dict[str, Any]) -> None:
    expected_checksum = geography_catalog_checksum()
    if payload.get("geography_catalog_checksum") != expected_checksum:
        raise ModelSerializationError(
            "Saved CTGAN geography catalog checksum is incompatible with the current deterministic catalog."
        )
    catalog_path = input_path / "geography_catalog.json"
    if not catalog_path.exists():
        raise ModelSerializationError("Missing geography_catalog.json for CTGAN geography_model_version=2.")
    with catalog_path.open(encoding="utf-8") as file:
        catalog_payload = json.load(file)
    if catalog_payload.get("checksum") != expected_checksum:
        raise ModelSerializationError("geography_catalog.json checksum does not match the current deterministic catalog.")
    keys = {entry.get("geo_key") for entry in catalog_payload.get("entries", []) if isinstance(entry, dict)}
    expected_keys = {entry.geo_key for entry in build_geography_catalog()}
    if keys != expected_keys:
        raise ModelSerializationError("geography_catalog.json does not contain the expected Geo_Key categories.")
