"""Centralized schema, domain, semantic, and identifier validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.domain.brazil import (
    STATE_DDDS,
    STATE_MUNICIPALITIES,
    region_for_state,
)
from synthetic_br_profiles_gan.generators.demographics import calcular_idade, parse_data_nascimento
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.validators.brazilian import (
    extrair_ddd,
    validar_cnh,
    validar_cpf,
    validar_formato_rg,
    validar_telefone,
    validar_titulo_eleitor,
)


@dataclass(frozen=True)
class ValidationResult:
    """Validation report plus row mask for internal candidate selection."""

    report: dict[str, Any]
    valid_mask: pd.Series


def _reference_date(value: str | date | datetime | None) -> date:
    if value is None:
        return datetime.now().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def validate_profile_dataframe(
    df: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
    final: bool = True,
    reference_date: str | date | datetime | None = None,
) -> ValidationResult:
    """Validate a model or final profile dataframe against project metadata."""
    metadata = metadata or default_metadata()
    required_columns = metadata.required_columns(final=final)
    valid_mask = pd.Series(True, index=df.index)
    counts: Counter[str] = Counter()
    details: dict[str, Any] = {"missing_columns": []}
    reference = _reference_date(reference_date)

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        counts["missing_required_columns"] = len(missing)
        details["missing_columns"] = missing
        valid_mask[:] = False

    present_required = [column for column in required_columns if column in df.columns]
    for column in present_required:
        null_mask = df[column].isna()
        if null_mask.any():
            counts["null_required_fields"] += int(null_mask.sum())
            valid_mask &= ~null_mask

    numeric_columns = [
        column
        for column in present_required
        if metadata.columns.get(column) and metadata.columns[column].kind in {"integer", "numeric"}
    ]
    for column in numeric_columns:
        column_meta = metadata.columns[column]
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_numeric = numeric.isna() | ~np.isfinite(numeric.astype(float))
        if invalid_numeric.any():
            counts["invalid_numeric_values"] += int(invalid_numeric.sum())
            valid_mask &= ~invalid_numeric
        if column_meta.min_value is not None:
            below = numeric < column_meta.min_value
            counts[f"{column}_below_min"] += int(below.sum())
            valid_mask &= ~below.fillna(True)
        if column_meta.max_value is not None:
            above = numeric > column_meta.max_value
            counts[f"{column}_above_max"] += int(above.sum())
            valid_mask &= ~above.fillna(True)
        if column_meta.kind == "integer":
            non_integer = numeric.dropna().mod(1).ne(0)
            if non_integer.any():
                invalid_index = non_integer[non_integer].index
                counts[f"{column}_non_integer"] += len(invalid_index)
                valid_mask.loc[invalid_index] = False

    for column in present_required:
        column_meta = metadata.columns.get(column)
        if not column_meta or not column_meta.categories:
            continue
        if column_meta.kind == "categorical":
            unexpected = ~df[column].astype(str).isin([str(category) for category in column_meta.categories])
        elif column_meta.discrete:
            numeric = pd.to_numeric(df[column], errors="coerce")
            unexpected = ~numeric.isin(column_meta.categories)
        else:
            continue
        if unexpected.any():
            counts[f"{column}_unexpected_category"] += int(unexpected.sum())
            valid_mask &= ~unexpected

    if {"Estado", "Regiao"}.issubset(df.columns):
        mismatch = df.apply(lambda row: region_for_state(str(row["Estado"])) != str(row["Regiao"]), axis=1)
        counts["estado_regiao_incompativel"] += int(mismatch.sum())
        valid_mask &= ~mismatch

    if {"Estado", "Municipio"}.issubset(df.columns):
        mismatch = df.apply(
            lambda row: str(row["Municipio"]) not in STATE_MUNICIPALITIES.get(str(row["Estado"]), ()),
            axis=1,
        )
        counts["municipio_estado_incompativel"] += int(mismatch.sum())
        valid_mask &= ~mismatch

    if {"Estado", "DDD"}.issubset(df.columns):
        ddd_numeric = pd.to_numeric(df["DDD"], errors="coerce")
        mismatch = df.apply(
            lambda row: int(row["DDD"]) not in STATE_DDDS.get(str(row["Estado"]), ())
            if pd.notna(row["DDD"])
            else True,
            axis=1,
        )
        mismatch |= ddd_numeric.isna()
        counts["ddd_estado_incompativel"] += int(mismatch.sum())
        valid_mask &= ~mismatch

    if final:
        _validate_final_fields(df, valid_mask, counts, reference)

    duplicated_rows = df.duplicated()
    if duplicated_rows.any():
        counts["duplicated_rows"] = int(duplicated_rows.sum())
        valid_mask &= ~duplicated_rows

    reason_counts = {key: int(value) for key, value in counts.items() if int(value) != 0}
    report = {
        "n_rows": int(len(df)),
        "invalid_rows": int((~valid_mask).sum()),
        "valid_rows": int(valid_mask.sum()),
        "reason_counts": reason_counts,
        "details": details,
        "is_valid": bool(valid_mask.all() and not missing),
    }
    return ValidationResult(report=report, valid_mask=valid_mask)


def _validate_final_fields(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    counts: Counter[str],
    reference: date,
) -> None:
    if {"Idade", "Data_Nascimento"}.issubset(df.columns):
        mismatch = []
        for _, row in df.iterrows():
            try:
                mismatch.append(calcular_idade(row["Data_Nascimento"], reference) != int(row["Idade"]))
            except (ValueError, TypeError):
                mismatch.append(True)
        mismatch_series = pd.Series(mismatch, index=df.index)
        counts["idade_data_nascimento_incompativel"] += int(mismatch_series.sum())
        valid_mask &= ~mismatch_series
    elif "Data_Nascimento" in df.columns:
        invalid = []
        for value in df["Data_Nascimento"]:
            try:
                parse_data_nascimento(value)
                invalid.append(False)
            except (ValueError, TypeError):
                invalid.append(True)
        invalid_series = pd.Series(invalid, index=df.index)
        counts["data_nascimento_invalida"] += int(invalid_series.sum())
        valid_mask &= ~invalid_series

    validators = {
        "CPF": validar_cpf,
        "CNH": validar_cnh,
        "RG": validar_formato_rg,
        "Titulo_Eleitor": validar_titulo_eleitor,
        "Telefone": validar_telefone,
    }
    for column, validator in validators.items():
        if column not in df.columns:
            continue
        invalid = ~df[column].astype(str).apply(validator)
        counts[f"{column}_invalido"] += int(invalid.sum())
        valid_mask &= ~invalid

    if {"Telefone", "DDD"}.issubset(df.columns):
        mismatch = df.apply(lambda row: extrair_ddd(row["Telefone"]) != int(row["DDD"]), axis=1)
        counts["telefone_ddd_incompativel"] += int(mismatch.sum())
        valid_mask &= ~mismatch

    for column in ["CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"]:
        if column not in df.columns:
            continue
        duplicates = df[column].duplicated()
        counts[f"{column}_duplicado"] += int(duplicates.sum())
        valid_mask &= ~duplicates


def validate_core_dataframe(
    df: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
) -> ValidationResult:
    """Validate only model columns before derived attributes are generated."""
    return validate_profile_dataframe(df, metadata=metadata, final=False)
