"""Dados de calibração controlados com dependências semânticas documentadas."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.config import ConfigDict, deep_merge
from synthetic_br_profiles_gan.domain.brazil import (
    REGION_STATES,
    REGIONS,
    STATE_DDDS,
    STATE_MUNICIPALITIES,
)
from synthetic_br_profiles_gan.metadata import (
    EDUCATION_CATEGORIES,
    GENDER_CATEGORIES,
    MARITAL_STATUS_CATEGORIES,
    MODEL_COLUMNS,
    OCCUPATION_CATEGORIES,
    DatasetMetadata,
    default_metadata,
)


DEFAULT_CALIBRATION_CONFIG: ConfigDict = {
    "seed": 41,
    "num_rows": 20000,
    "holdout_fraction": 0.2,
    "income": {"min": 800.0, "max": 50000.0},
    "age": {"min": 18, "max": 85},
    "region_weights": {
        "Norte": 0.09,
        "Nordeste": 0.27,
        "Centro-Oeste": 0.08,
        "Sudeste": 0.42,
        "Sul": 0.14,
    },
}

REGION_INCOME_MULTIPLIER = {
    "Norte": 0.90,
    "Nordeste": 0.86,
    "Centro-Oeste": 1.06,
    "Sudeste": 1.14,
    "Sul": 1.08,
}

EDUCATION_INCOME_MULTIPLIER = {
    "Fundamental": 0.74,
    "Ensino Medio": 0.92,
    "Superior Incompleto": 1.08,
    "Superior Completo": 1.55,
    "Pos-graduacao": 2.05,
}

OCCUPATION_INCOME_MULTIPLIER = {
    "Estudante": 0.46,
    "Servicos Gerais": 0.74,
    "Tecnico": 1.02,
    "Analista": 1.32,
    "Coordenador": 1.62,
    "Gerente": 2.18,
    "Autonomo": 1.08,
    "Aposentado": 0.88,
}


def _choice(rng: np.random.Generator, options: list[Any] | tuple[Any, ...], weights: list[float] | None = None) -> Any:
    values = list(options)
    if weights is None:
        index = int(rng.integers(0, len(values)))
        return values[index]
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    index = int(rng.choice(len(values), p=probabilities))
    return values[index]


def _sample_age(rng: np.random.Generator, min_age: int, max_age: int) -> int:
    bucket = _choice(rng, ["young", "adult", "mature", "senior"], [0.24, 0.39, 0.27, 0.10])
    if bucket == "young":
        value = rng.triangular(min_age, 25, 34)
    elif bucket == "adult":
        value = rng.triangular(28, 40, 54)
    elif bucket == "mature":
        value = rng.triangular(45, 56, 70)
    else:
        value = rng.triangular(61, 68, max_age)
    return int(np.clip(round(value), min_age, max_age))


def _sample_education(rng: np.random.Generator, age: int) -> str:
    if age < 22:
        weights = [0.10, 0.45, 0.36, 0.08, 0.01]
    elif age < 30:
        weights = [0.08, 0.34, 0.22, 0.29, 0.07]
    elif age < 55:
        weights = [0.14, 0.39, 0.08, 0.29, 0.10]
    else:
        weights = [0.28, 0.38, 0.02, 0.24, 0.08]
    return str(_choice(rng, EDUCATION_CATEGORIES, weights))


def _sample_occupation(rng: np.random.Generator, age: int, education: str) -> str:
    if age >= 65:
        return str(_choice(rng, OCCUPATION_CATEGORIES, [0.01, 0.05, 0.07, 0.08, 0.04, 0.03, 0.12, 0.60]))
    if age <= 23 and education in {"Ensino Medio", "Superior Incompleto"}:
        return str(_choice(rng, OCCUPATION_CATEGORIES, [0.40, 0.14, 0.16, 0.12, 0.02, 0.00, 0.16, 0.00]))
    if education == "Fundamental":
        weights = [0.02, 0.50, 0.16, 0.03, 0.01, 0.00, 0.26, 0.02]
    elif education == "Ensino Medio":
        weights = [0.03, 0.28, 0.30, 0.12, 0.04, 0.01, 0.20, 0.02]
    elif education == "Superior Incompleto":
        weights = [0.10, 0.08, 0.28, 0.28, 0.07, 0.02, 0.16, 0.01]
    elif education == "Superior Completo":
        weights = [0.01, 0.03, 0.20, 0.42, 0.15, 0.06, 0.12, 0.01]
    else:
        weights = [0.00, 0.01, 0.08, 0.44, 0.20, 0.16, 0.10, 0.01]
    return str(_choice(rng, OCCUPATION_CATEGORIES, weights))


def _sample_marital_status(rng: np.random.Generator, age: int) -> str:
    if age < 25:
        weights = [0.82, 0.08, 0.08, 0.02, 0.00]
    elif age < 40:
        weights = [0.38, 0.34, 0.18, 0.09, 0.01]
    elif age < 65:
        weights = [0.18, 0.50, 0.12, 0.17, 0.03]
    else:
        weights = [0.12, 0.44, 0.06, 0.16, 0.22]
    return str(_choice(rng, MARITAL_STATUS_CATEGORIES, weights))


def _sample_dependents(rng: np.random.Generator, age: int, marital_status: str) -> int:
    if age < 24:
        weights = [0.84, 0.13, 0.03, 0.00, 0.00, 0.00, 0.00]
    elif marital_status in {"Casado", "Uniao Estavel"} and 28 <= age <= 55:
        weights = [0.20, 0.25, 0.30, 0.16, 0.06, 0.02, 0.01]
    elif marital_status == "Solteiro":
        weights = [0.66, 0.20, 0.10, 0.03, 0.01, 0.00, 0.00]
    elif age >= 60:
        weights = [0.45, 0.24, 0.18, 0.08, 0.03, 0.01, 0.01]
    else:
        weights = [0.32, 0.26, 0.24, 0.12, 0.04, 0.01, 0.01]
    return int(_choice(rng, list(range(7)), weights))


def _sample_income(
    rng: np.random.Generator,
    age: int,
    education: str,
    occupation: str,
    region: str,
    minimum: float,
    maximum: float,
) -> float:
    age_curve = 0.72 + min(age, 58) / 58
    if age >= 67:
        age_curve *= 0.88
    log_noise = float(rng.lognormal(mean=math.log(2100), sigma=0.55))
    mixture_boost = float(rng.lognormal(mean=0.0, sigma=0.32)) if rng.random() < 0.12 else 1.0
    income = (
        log_noise
        * age_curve
        * EDUCATION_INCOME_MULTIPLIER[education]
        * OCCUPATION_INCOME_MULTIPLIER[occupation]
        * REGION_INCOME_MULTIPLIER[region]
        * mixture_boost
    )
    return round(float(np.clip(income, minimum, maximum)), 2)


def generate_calibration_dataset(
    num_rows: int | None = None,
    seed: int | None = None,
    config: ConfigDict | None = None,
) -> pd.DataFrame:
    """Gera uma base de calibração sintética com dependências semânticas."""
    effective = deep_merge(DEFAULT_CALIBRATION_CONFIG, config or {})
    if num_rows is None:
        num_rows = int(effective["num_rows"])
    if seed is None:
        seed = int(effective["seed"])

    rng = np.random.default_rng(seed)
    income_min = float(effective["income"]["min"])
    income_max = float(effective["income"]["max"])
    min_age = int(effective["age"]["min"])
    max_age = int(effective["age"]["max"])
    region_weights = effective["region_weights"]
    regions = list(REGIONS)
    weights = [float(region_weights[region]) for region in regions]

    rows: list[dict[str, Any]] = []
    for _ in range(int(num_rows)):
        region = str(_choice(rng, regions, weights))
        state = str(_choice(rng, REGION_STATES[region]))
        city = str(_choice(rng, STATE_MUNICIPALITIES[state]))
        ddd = int(_choice(rng, STATE_DDDS[state]))
        age = _sample_age(rng, min_age, max_age)
        gender = str(_choice(rng, GENDER_CATEGORIES, [0.505, 0.485, 0.010]))
        education = _sample_education(rng, age)
        occupation = _sample_occupation(rng, age, education)
        marital_status = _sample_marital_status(rng, age)
        dependents = _sample_dependents(rng, age, marital_status)
        income = _sample_income(rng, age, education, occupation, region, income_min, income_max)
        rows.append(
            {
                "Idade": age,
                "Genero": gender,
                "Regiao": region,
                "Estado": state,
                "Municipio": city,
                "Escolaridade": education,
                "Estado_Civil": marital_status,
                "Ocupacao": occupation,
                "Renda": income,
                "Dependentes": dependents,
                "DDD": ddd,
            }
        )

    df = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    return coerce_model_dtypes(df, default_metadata())


def coerce_model_dtypes(df: pd.DataFrame, metadata: DatasetMetadata | None = None) -> pd.DataFrame:
    """Converte colunas canônicas do modelo para dtypes estáveis do pandas."""
    coerced = df.copy()
    metadata = metadata or default_metadata()
    for column in metadata.model_columns:
        if column not in coerced.columns:
            continue
        column_meta = metadata.columns[column]
        if column_meta.kind == "integer":
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce").round().astype("Int64")
        elif column_meta.kind == "numeric":
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype(float)
        elif column_meta.kind == "categorical":
            coerced[column] = coerced[column].astype("string")
    return coerced


def split_train_holdout(
    df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 41,
    train_rows: int | None = None,
    holdout_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide dados de calibração em treino e holdout sem vazamento.

    Quando ``train_rows`` e ``holdout_rows`` são fornecidos, a divisão usa esses
    tamanhos exatos após embaralhamento determinístico. O comportamento baseado
    em fração é preservado para chamadas existentes.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if (train_rows is None) != (holdout_rows is None):
        raise ValueError("train_rows and holdout_rows must be provided together.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    if train_rows is not None and holdout_rows is not None:
        train_size = int(train_rows)
        holdout_size = int(holdout_rows)
        if train_size <= 0 or holdout_size <= 0:
            raise ValueError("train_rows and holdout_rows must be greater than zero.")
        if train_size + holdout_size > len(df):
            raise ValueError("Requested split sizes exceed the number of calibration rows.")
        train_idx = indices[:train_size]
        holdout_idx = indices[train_size : train_size + holdout_size]
    else:
        holdout_size = max(1, int(round(len(df) * holdout_fraction)))
        holdout_idx = indices[:holdout_size]
        train_idx = indices[holdout_size:]
    train = df.iloc[train_idx].reset_index(drop=True)
    holdout = df.iloc[holdout_idx].reset_index(drop=True)
    return train, holdout


def save_calibration_splits(
    df: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    output_dir: str | Path,
    metadata: DatasetMetadata | None = None,
) -> dict[str, Path]:
    """Persiste artefatos de calibração, treino, holdout e metadados."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "calibration": output_path / "calibration.parquet",
        "train": output_path / "train.parquet",
        "holdout": output_path / "holdout.parquet",
        "metadata": output_path / "metadata.json",
    }
    df.to_parquet(paths["calibration"], index=False)
    train.to_parquet(paths["train"], index=False)
    holdout.to_parquet(paths["holdout"], index=False)
    (metadata or default_metadata()).save(paths["metadata"])
    return paths
