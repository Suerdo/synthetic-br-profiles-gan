"""Métricas de validade relacional para as colunas-base."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.domain.occupations import get_occupation_profile
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata


def relational_validity_report(frame: pd.DataFrame, metadata: DatasetMetadata | None = None) -> dict[str, Any]:
    """Calcula validade não relacional e profissional sem alterar os dados."""
    metadata = metadata or default_metadata()
    data = frame.reset_index(drop=True)
    total = int(len(data))
    if total == 0:
        return {
            "total_rows": 0,
            "validity": {
                "non_relational_validity_rate": 0.0,
                "professional_validity_rate": 0.0,
                "occupation_education_valid_rate": 0.0,
                "occupation_age_valid_rate": 0.0,
            },
            "invalid_counts": {},
            "top_invalid_combinations": {},
            "interpretation": (
                "Métricas calculadas nas colunas-base. Relações geográficas são reportadas "
                "separadamente em evaluation.geography."
            ),
        }

    known = _known_category_mask(data, metadata)
    age = _numeric_domain_mask(data, metadata, "Idade", integer=True)
    income = _numeric_domain_mask(data, metadata, "Renda", integer=False)
    dependents = _numeric_domain_mask(data, metadata, "Dependentes", integer=True)
    professional_categories = _known_category_mask(data, metadata, columns=("Escolaridade", "Ocupacao"))
    occupation_education = _occupation_education_mask(data)
    occupation_age = _occupation_age_mask(data)
    non_relational = known & age & income & dependents
    professional = professional_categories & age & income & occupation_education & occupation_age
    masks = {
        "known_categories": known,
        "age_domain": age,
        "income_domain": income,
        "dependents_domain": dependents,
        "occupation_education": occupation_education,
        "occupation_age": occupation_age,
        "non_relational_joint": non_relational,
        "professional_joint": professional,
    }
    return {
        "total_rows": total,
        "validity": {
            "non_relational_validity_rate": _rate(int(non_relational.sum()), total),
            "professional_validity_rate": _rate(int(professional.sum()), total),
            "occupation_education_valid_rate": _rate(int(occupation_education.sum()), total),
            "occupation_age_valid_rate": _rate(int(occupation_age.sum()), total),
        },
        "invalid_counts": {name: int(total - int(mask.sum())) for name, mask in masks.items()},
        "top_invalid_combinations": {
            "occupation_education": _top_invalid_pairs(data, ~occupation_education, ("Escolaridade", "Ocupacao")),
            "occupation_age": _top_invalid_pairs(data, ~occupation_age, ("Idade", "Ocupacao")),
            "known_categories": _top_unknown_categories(data, metadata, ~known),
        },
        "interpretation": (
            "A validade não relacional considera categorias conhecidas e domínios individuais de idade, "
            "renda e dependentes. A validade profissional considera escolaridade, idade, ocupação e renda. "
            "Relações geográficas são reportadas separadamente em evaluation.geography."
        ),
    }


def _known_category_mask(
    frame: pd.DataFrame,
    metadata: DatasetMetadata,
    columns: tuple[str, ...] | None = None,
) -> pd.Series:
    target_columns = columns or tuple(
        column
        for column in metadata.model_columns
        if column in frame.columns
        and metadata.columns[column].categories
        and (metadata.columns[column].kind == "categorical" or metadata.columns[column].discrete)
    )
    mask = pd.Series(True, index=frame.index)
    for column in target_columns:
        meta = metadata.columns[column]
        if meta.discrete:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.isin(meta.categories)
        else:
            valid = frame[column].astype(str).isin([str(category) for category in meta.categories])
        mask &= valid.fillna(False).astype(bool)
    return mask


def _numeric_domain_mask(frame: pd.DataFrame, metadata: DatasetMetadata, column: str, *, integer: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    meta = metadata.columns[column]
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = values.notna()
    if meta.min_value is not None:
        mask &= values.ge(float(meta.min_value))
    if meta.max_value is not None:
        mask &= values.le(float(meta.max_value))
    if integer:
        mask &= values.dropna().mod(1).reindex(frame.index, fill_value=1).eq(0)
    return mask.fillna(False).astype(bool)


def _occupation_education_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"Ocupacao", "Escolaridade"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)

    def valid(row: pd.Series) -> bool:
        profile = get_occupation_profile(str(row["Ocupacao"]))
        return bool(profile is not None and str(row["Escolaridade"]) in profile.allowed_education)

    return frame.apply(valid, axis=1).astype(bool)


def _occupation_age_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"Ocupacao", "Idade"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)

    def valid(row: pd.Series) -> bool:
        profile = get_occupation_profile(str(row["Ocupacao"]))
        if profile is None:
            return False
        try:
            age = int(row["Idade"])
        except (TypeError, ValueError):
            return False
        return bool(age >= profile.minimum_age and (profile.maximum_age is None or age <= profile.maximum_age))

    return frame.apply(valid, axis=1).astype(bool)


def _top_invalid_pairs(frame: pd.DataFrame, invalid_mask: pd.Series, columns: tuple[str, ...], limit: int = 10) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    invalid = frame.loc[invalid_mask.astype(bool), available].copy()
    if invalid.empty:
        return []
    counts = invalid.astype(str).value_counts().head(int(limit))
    rows: list[dict[str, Any]] = []
    for key, count in counts.items():
        values = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(available, values)), "count": int(count)})
    return rows


def _top_unknown_categories(
    frame: pd.DataFrame,
    metadata: DatasetMetadata,
    invalid_mask: pd.Series,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not invalid_mask.any():
        return []
    rows: list[dict[str, Any]] = []
    subset = frame.loc[invalid_mask.astype(bool)].copy()
    for column in metadata.model_columns:
        if column not in subset.columns or not metadata.columns[column].categories:
            continue
        meta = metadata.columns[column]
        if meta.discrete:
            valid = pd.to_numeric(subset[column], errors="coerce").isin(meta.categories)
        else:
            valid = subset[column].astype(str).isin([str(category) for category in meta.categories])
        for value, count in Counter(subset.loc[~valid.fillna(False), column].astype(str)).most_common(limit):
            rows.append({"column": column, "value": value, "count": int(count)})
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:limit]


def _rate(valid: int, total: int) -> float:
    return 0.0 if int(total) <= 0 else float(int(valid) / int(total))
