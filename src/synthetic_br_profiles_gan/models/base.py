"""Protocolo comum dos sintetizadores e fábrica de modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

import pandas as pd

from synthetic_br_profiles_gan.config import ConfigDict
from synthetic_br_profiles_gan.metadata import DatasetMetadata


class TabularSynthesizer(Protocol):
    """Interface comum para todos os sintetizadores tabulares."""

    model_name: str

    def fit(self, data: pd.DataFrame, metadata: DatasetMetadata) -> None:
        """Ajusta o sintetizador usando somente os dados de treinamento."""

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Amostra linhas sintéticas no schema do modelo."""

    def save(self, output_path: Path) -> None:
        """Salva o estado do sintetizador."""

    @classmethod
    def load(cls, input_path: Path) -> Self:
        """Carrega o estado de um sintetizador."""


def create_synthesizer(model_name: str, config: ConfigDict | None = None) -> TabularSynthesizer:
    """Cria uma implementação de sintetizador pelo nome."""
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"programmatic", "programmatic_synthesizer"}:
        from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer

        return ProgrammaticSynthesizer(config=config)
    if normalized in {"simple_gan", "simple_tabular_gan", "dense_tabular_gan"}:
        from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN

        return SimpleTabularGAN(config=config)
    if normalized in {"ctgan", "ctgan_synthesizer"}:
        from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

        return CTGANSynthesizer(config=config)
    raise ValueError(f"Unknown synthesizer model: {model_name}")
