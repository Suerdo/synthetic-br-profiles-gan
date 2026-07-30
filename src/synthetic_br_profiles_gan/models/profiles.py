"""Perfis nomeados de configurações candidatas de sintetizadores."""

from __future__ import annotations

import importlib.metadata
from copy import deepcopy
from typing import Any

from synthetic_br_profiles_gan.domain.geography import GEOGRAPHY_MODEL_VERSION, LEGACY_GEOGRAPHY_MODEL_VERSION
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, INCOME_MODEL_VERSION


CTGAN_INCOME_V3_RECOMMENDED_CANDIDATE: dict[str, Any] = {
    "profile_name": "ctgan_income_v3_recommended_candidate",
    "model": "ctgan",
    "purpose": "recommended_candidate",
    "seed_policy": "definida por execução; a seed não é congelada no perfil",
    "ctgan": {
        "embedding_dim": None,
        "generator_dim": None,
        "discriminator_dim": None,
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "generator_decay": 0.000001,
        "discriminator_decay": 0.000001,
        "discriminator_steps": 1,
        "log_frequency": False,
        "pac": 10,
        "epochs": 20,
        "batch_size": 500,
        "verbose": False,
        "enable_gpu": False,
        "cuda": None,
    },
    "categorical_vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
    "income_model_version": INCOME_MODEL_VERSION,
    "geography_model_version": LEGACY_GEOGRAPHY_MODEL_VERSION,
}

CTGAN_INCOME_V3_GEO_V2_CANDIDATE: dict[str, Any] = deepcopy(CTGAN_INCOME_V3_RECOMMENDED_CANDIDATE)
CTGAN_INCOME_V3_GEO_V2_CANDIDATE["profile_name"] = "ctgan_income_v3_geo_v2_candidate"
CTGAN_INCOME_V3_GEO_V2_CANDIDATE["geography_model_version"] = GEOGRAPHY_MODEL_VERSION
CTGAN_INCOME_V3_GEO_V2_CANDIDATE["ctgan"] = {
    **CTGAN_INCOME_V3_GEO_V2_CANDIDATE["ctgan"],
    "geography_model_version": GEOGRAPHY_MODEL_VERSION,
}


def ctgan_income_v3_recommended_candidate_profile() -> dict[str, Any]:
    """Retorna o perfil congelado da CTGAN candidata para renda v3."""
    profile = deepcopy(CTGAN_INCOME_V3_RECOMMENDED_CANDIDATE)
    try:
        profile["ctgan_version"] = importlib.metadata.version("ctgan")
    except importlib.metadata.PackageNotFoundError:
        profile["ctgan_version"] = None
    return profile


def ctgan_income_v3_geo_v2_candidate_profile() -> dict[str, Any]:
    """Retorna o perfil CTGAN com renda v3 e representação geográfica composta."""
    profile = deepcopy(CTGAN_INCOME_V3_GEO_V2_CANDIDATE)
    try:
        profile["ctgan_version"] = importlib.metadata.version("ctgan")
    except importlib.metadata.PackageNotFoundError:
        profile["ctgan_version"] = None
    return profile
