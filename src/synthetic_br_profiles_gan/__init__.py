"""Pipeline de geração e avaliação de perfis sintéticos brasileiros."""

from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.pipeline import run_pipeline

__version__ = "0.2.0"

__all__ = ["DatasetMetadata", "default_metadata", "run_pipeline"]

