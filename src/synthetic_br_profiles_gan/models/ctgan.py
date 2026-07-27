"""CTGAN synthesizer wrapper using the standalone ctgan package."""

from __future__ import annotations

import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.calibration import coerce_model_dtypes
from synthetic_br_profiles_gan.config import ConfigDict
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
}


class CTGANSynthesizer:
    """Real CTGAN implementation backed by the standalone ``ctgan`` package."""

    model_name = "ctgan"

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = {**DEFAULT_CTGAN_CONFIG, **(config or {})}
        self.metadata: DatasetMetadata = default_metadata()
        self.model: Any | None = None
        self.discrete_columns: list[str] = []
        self.library_version: str | None = None

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
        """Fit CTGAN, declaring categorical/discrete columns explicitly."""
        set_global_seed(int(self.config.get("seed", 41)), seed_tensorflow=False, seed_torch=True)
        CTGAN = self._ctgan_class()
        self.metadata = metadata
        self.discrete_columns = metadata.categorical_columns(include_discrete_numeric=True)
        self.library_version = importlib.metadata.version("ctgan")
        train = coerce_model_dtypes(data[metadata.model_columns], metadata)
        model_kwargs = {
            "epochs": int(self.config["epochs"]),
            "batch_size": int(self.config["batch_size"]),
            "verbose": bool(self.config["verbose"]),
            "enable_gpu": bool(self.config.get("enable_gpu", self.config.get("cuda", False))),
        }
        self.model = CTGAN(**model_kwargs)
        self.model.fit(train, self.discrete_columns)

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Sample canonical model rows and restore configured dtypes."""
        if self.model is None:
            raise SyntheticModelError("CTGANSynthesizer must be fitted or loaded before sampling.")
        sampled = self.model.sample(int(num_rows))
        sampled = sampled[self.metadata.model_columns]
        return coerce_model_dtypes(sampled, self.metadata)

    def save(self, output_path: Path) -> None:
        """Save CTGAN model, metadata, and fit configuration."""
        if self.model is None:
            raise SyntheticModelError("Cannot save an unfitted CTGANSynthesizer.")
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "model.pkl").open("wb") as file:
            pickle.dump(self.model, file)
        self.metadata.save(output_path / "metadata.json")
        payload = {
            "config": self.config,
            "discrete_columns": self.discrete_columns,
            "library": "ctgan",
            "library_version": self.library_version or importlib.metadata.version("ctgan"),
        }
        with (output_path / "metadata_ctgan.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "CTGANSynthesizer":
        """Load a saved CTGAN synthesizer."""
        cls._ctgan_class()
        try:
            with (input_path / "metadata_ctgan.json").open(encoding="utf-8") as file:
                payload = json.load(file)
            instance = cls(config=payload.get("config", {}))
            instance.discrete_columns = list(payload.get("discrete_columns", []))
            instance.library_version = payload.get("library_version")
            instance.metadata = DatasetMetadata.load(input_path / "metadata.json")
            with (input_path / "model.pkl").open("rb") as file:
                instance.model = pickle.load(file)
            return instance
        except Exception as exc:
            raise ModelSerializationError(f"Could not load CTGANSynthesizer from {input_path}: {exc}") from exc
