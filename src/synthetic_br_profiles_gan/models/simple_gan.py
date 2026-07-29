"""Adaptador que expõe a GAN densa em Keras pelo protocolo de sintetizadores.

Os detalhes de implementação ficam em ``models.gan``; este módulo concentra o
contrato comum fit/sample/save/load usado pelo pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.config import ConfigDict
from synthetic_br_profiles_gan.exceptions import ModelSerializationError, SyntheticModelError
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.models.gan import (
    _require_tensorflow,
    build_discriminator,
    build_gan,
    build_generator,
    train_gan,
)
from synthetic_br_profiles_gan.models.preprocessing import DataPreprocessor


DEFAULT_SIMPLE_GAN_CONFIG: ConfigDict = {
    "seed": 41,
    "latent_dim": 16,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "generator_learning_rate": None,
    "discriminator_learning_rate": None,
    "beta_1": 0.5,
    "verbose_every": 10,
    "metrics_every": 10,
    "generator_hidden_dims": None,
    "discriminator_hidden_dims": None,
    "discriminator_dropout": 0.0,
    "generator_batch_norm": False,
    "label_smoothing": 0.0,
    "discriminator_steps": 1,
}


class SimpleTabularGAN:
    """Baseline de GAN tabular densa preservado da implementação original."""

    model_name = "simple_gan"

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = {**DEFAULT_SIMPLE_GAN_CONFIG, **(config or {})}
        self.metadata: DatasetMetadata = default_metadata()
        self.preprocessor: DataPreprocessor | None = None
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.training_history: dict[str, Any] = {}
        self._sample_calls = 0

    def fit(self, data: pd.DataFrame, metadata: DatasetMetadata) -> None:
        """Ajusta a GAN densa usando somente o split de treinamento."""
        _, _, Adam = _require_tensorflow()
        self.metadata = metadata
        self.preprocessor = DataPreprocessor(metadata=metadata)
        encoded = self.preprocessor.fit_transform(data[metadata.model_columns])
        latent_dim = int(self.config["latent_dim"])
        generator_learning_rate = float(self.config.get("generator_learning_rate") or self.config["learning_rate"])
        discriminator_learning_rate = float(self.config.get("discriminator_learning_rate") or self.config["learning_rate"])
        self.generator = build_generator(
            latent_dim,
            self.preprocessor.output_dim,
            hidden_dims=self.config.get("generator_hidden_dims"),
            batch_normalization=bool(self.config.get("generator_batch_norm", False)),
        )
        self.discriminator = build_discriminator(
            self.preprocessor.output_dim,
            hidden_dims=self.config.get("discriminator_hidden_dims"),
            dropout=float(self.config.get("discriminator_dropout", 0.0)),
        )
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=discriminator_learning_rate, beta_1=float(self.config["beta_1"])),
            metrics=["accuracy"],
        )
        self.gan = build_gan(
            self.generator,
            self.discriminator,
            latent_dim,
            learning_rate=generator_learning_rate,
            beta_1=float(self.config["beta_1"]),
        )

        def sample_metrics(sample: np.ndarray) -> dict[str, float]:
            return {
                "encoded_mean": float(np.mean(sample)),
                "encoded_std": float(np.std(sample)),
                "encoded_min": float(np.min(sample)),
                "encoded_max": float(np.max(sample)),
            }

        self.training_history = train_gan(
            generator=self.generator,
            discriminator=self.discriminator,
            gan=self.gan,
            data=encoded,
            latent_dim=latent_dim,
            epochs=int(self.config["epochs"]),
            batch_size=int(self.config["batch_size"]),
            verbose_every=int(self.config["verbose_every"]),
            seed=int(self.config["seed"]),
            metrics_every=int(self.config["metrics_every"]),
            sample_metric_fn=sample_metrics,
            label_smoothing=float(self.config.get("label_smoothing", 0.0)),
            discriminator_steps=int(self.config.get("discriminator_steps", 1)),
        )
        self.training_history["config"] = self.config

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Amostra linhas canônicas do modelo sem filtro por limiar do discriminador."""
        if self.generator is None or self.preprocessor is None:
            raise SyntheticModelError("SimpleTabularGAN must be fitted or loaded before sampling.")
        rng = np.random.default_rng(int(self.config.get("seed", 41)) + self._sample_calls)
        self._sample_calls += 1
        latent_dim = int(self.config["latent_dim"])
        noise = rng.normal(0, 1, (int(num_rows), latent_dim))
        encoded = self.generator.predict(noise, verbose=0)
        sampled = self.preprocessor.inverse_transform(encoded)
        return sampled[self.metadata.model_columns]

    def save(self, output_path: Path) -> None:
        """Salva modelos Keras, pré-processador, metadados, configuração e histórico de treino."""
        if self.generator is None or self.discriminator is None or self.preprocessor is None:
            raise SyntheticModelError("Cannot save an unfitted SimpleTabularGAN.")
        output_path.mkdir(parents=True, exist_ok=True)
        self.generator.save(output_path / "generator.keras")
        self.discriminator.save(output_path / "discriminator.keras")
        self.preprocessor.save(output_path / "preprocessor.pkl")
        self.metadata.save(output_path / "metadata.json")
        with (output_path / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.config, file, ensure_ascii=False, indent=2)
        with (output_path / "training_history.json").open("w", encoding="utf-8") as file:
            json.dump(self.training_history, file, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, input_path: Path) -> "SimpleTabularGAN":
        """Carrega uma SimpleTabularGAN salva."""
        _, _, _ = _require_tensorflow()
        from tensorflow.keras.models import load_model

        try:
            with (input_path / "config.json").open(encoding="utf-8") as file:
                config = json.load(file)
            instance = cls(config=config)
            instance.metadata = DatasetMetadata.load(input_path / "metadata.json")
            instance.preprocessor = DataPreprocessor.load(input_path / "preprocessor.pkl")
            instance.generator = load_model(input_path / "generator.keras")
            instance.discriminator = load_model(input_path / "discriminator.keras")
            history_path = input_path / "training_history.json"
            if history_path.exists():
                with history_path.open(encoding="utf-8") as file:
                    instance.training_history = json.load(file)
            return instance
        except Exception as exc:
            raise ModelSerializationError(f"Could not load SimpleTabularGAN from {input_path}: {exc}") from exc
