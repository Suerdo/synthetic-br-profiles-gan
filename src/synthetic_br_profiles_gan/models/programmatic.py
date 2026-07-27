"""Purely programmatic baseline synthesizer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synthetic_br_profiles_gan.calibration import DEFAULT_CALIBRATION_CONFIG, generate_calibration_dataset
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge
from synthetic_br_profiles_gan.exceptions import ModelSerializationError
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata


class ProgrammaticSynthesizer:
    """Generate canonical model rows using the explicit calibration rules."""

    model_name = "programmatic"

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = deep_merge(DEFAULT_CALIBRATION_CONFIG, config or {})
        self.metadata: DatasetMetadata = default_metadata()
        self._sample_calls = 0

    def fit(self, data: pd.DataFrame, metadata: DatasetMetadata) -> None:
        """Store metadata; no training is required for the programmatic baseline."""
        self.metadata = metadata

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Sample rows from the same controlled rules used for calibration."""
        base_seed = int(self.config.get("seed", 41))
        seed = base_seed + self._sample_calls
        self._sample_calls += 1
        return generate_calibration_dataset(num_rows=num_rows, seed=seed, config=self.config)

    def save(self, output_path: Path) -> None:
        """Save programmatic configuration and metadata."""
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config, file, ensure_ascii=False, indent=2)
        self.metadata.save(output_path / "metadata.json")

    @classmethod
    def load(cls, input_path: Path) -> "ProgrammaticSynthesizer":
        """Load a saved programmatic baseline."""
        try:
            with (input_path / "config.json").open(encoding="utf-8") as file:
                config = json.load(file)
            instance = cls(config=config)
            metadata_path = input_path / "metadata.json"
            if metadata_path.exists():
                instance.metadata = DatasetMetadata.load(metadata_path)
            return instance
        except Exception as exc:
            raise ModelSerializationError(f"Could not load ProgrammaticSynthesizer from {input_path}: {exc}") from exc
