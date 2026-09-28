"""Compatibilidade para o catálogo de modelos usado pela interface Streamlit."""

from __future__ import annotations

from synthetic_br_profiles_gan.services.model_catalog import (
    MODEL_CATALOG,
    ModelCatalogEntry,
    model_catalog,
    model_catalog_by_name,
)

__all__ = [
    "MODEL_CATALOG",
    "ModelCatalogEntry",
    "model_catalog",
    "model_catalog_by_name",
]
