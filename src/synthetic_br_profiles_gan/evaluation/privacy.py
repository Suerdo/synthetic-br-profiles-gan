"""Indicadores de diversidade e risco de privacidade para dados sintéticos."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.localization import UNICODE_NORMALIZATION, normalize_text_value
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata


PARTIAL_MATCH_COLUMN_SETS: dict[str, list[str]] = {
    "demografia_localizacao": ["Genero", "Idade", "Regiao", "Estado", "Municipio"],
    "perfil_socioeconomico": ["Escolaridade", "Ocupacao", "Renda", "Dependentes"],
    "quase_identificadores_categoricos": [
        "Genero",
        "Regiao",
        "Estado",
        "Municipio",
        "Escolaridade",
        "Estado_Civil",
        "Ocupacao",
        "DDD",
    ],
}


def duplicate_row_rate(df: pd.DataFrame) -> float:
    """Retorna a taxa de linhas duplicadas."""
    if len(df) == 0:
        return 0.0
    return float(df.duplicated().sum() / len(df))


def exact_match_rate(synthetic: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> float:
    """Retorna a taxa de match exato com a referência nas colunas selecionadas."""
    return float(exact_match_metrics(synthetic, reference, columns)["exact_match_rate"])


def duplicate_base_row_metrics(
    synthetic: pd.DataFrame,
    columns: list[str],
    *,
    max_groups: int = 20,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Calcula duplicidade estruturada sobre combinações-base canônicas."""
    usable_columns = [column for column in columns if column in synthetic.columns]
    canonical = _canonical_frame(synthetic, usable_columns)
    total_rows = int(len(canonical))
    if total_rows == 0 or not usable_columns:
        return {
            "columns_used": usable_columns,
            "total_rows": total_rows,
            "unique_rows": 0,
            "duplicated_occurrences": 0,
            "duplicated_groups": 0,
            "duplicate_row_rate": 0.0,
            "unique_combination_rate": 0.0,
            "largest_duplicate_group": 0,
            "rows_in_duplicate_groups": 0,
            "duplicate_group_size_distribution": {},
            "largest_groups": [],
            "canonical_representation": _canonical_representation_description(usable_columns),
        }

    hashes = _row_hashes(canonical, usable_columns)
    counts = hashes.value_counts(sort=False)
    duplicate_counts = counts[counts > 1].sort_values(ascending=False)
    duplicated_occurrences = int((counts - 1).clip(lower=0).sum())
    duplicated_groups = int(len(duplicate_counts))
    rows_in_duplicate_groups = int(duplicate_counts.sum()) if duplicated_groups else 0
    largest_duplicate_group = int(duplicate_counts.iloc[0]) if duplicated_groups else 0
    distribution = duplicate_counts.value_counts().sort_index()
    groups: list[dict[str, Any]] = []
    for row_hash, occurrences in duplicate_counts.head(int(max_groups)).items():
        indices = [int(index) for index in hashes.index[hashes == row_hash].tolist()]
        payload: dict[str, Any] = {
            "row_hash": str(row_hash),
            "occurrences": int(occurrences),
            "indices": indices,
            "columns_used": usable_columns,
        }
        if model is not None:
            payload["model"] = model
        if seed is not None:
            payload["seed"] = int(seed)
        groups.append(payload)
    return {
        "columns_used": usable_columns,
        "total_rows": total_rows,
        "unique_rows": int(counts.shape[0]),
        "duplicated_occurrences": duplicated_occurrences,
        "duplicated_groups": duplicated_groups,
        "duplicate_row_rate": float(duplicated_occurrences / total_rows),
        "unique_combination_rate": float(counts.shape[0] / total_rows),
        "largest_duplicate_group": largest_duplicate_group,
        "rows_in_duplicate_groups": rows_in_duplicate_groups,
        "duplicate_group_size_distribution": {str(int(size)): int(amount) for size, amount in distribution.items()},
        "largest_groups": groups,
        "canonical_representation": _canonical_representation_description(usable_columns),
    }


def exact_match_metrics(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str],
    *,
    max_hashes: int = 100,
) -> dict[str, Any]:
    """Calcula correspondências exatas canônicas sem expor valores das linhas."""
    usable_columns = [column for column in columns if column in synthetic.columns and column in reference.columns]
    synthetic_rows = int(len(synthetic))
    reference_rows = int(len(reference))
    if synthetic_rows == 0 or reference_rows == 0 or not usable_columns:
        return {
            "columns_used": usable_columns,
            "synthetic_rows": synthetic_rows,
            "reference_rows": reference_rows,
            "exact_match_count": 0,
            "exact_match_rate": 0.0,
            "distinct_matched_synthetic_rows": 0,
            "distinct_reference_rows_matched": 0,
            "matched_row_hashes": [],
            "canonical_representation": _canonical_representation_description(usable_columns),
        }

    synthetic_hashes = _row_hashes(_canonical_frame(synthetic, usable_columns), usable_columns)
    reference_hashes = _row_hashes(_canonical_frame(reference, usable_columns), usable_columns)
    reference_set = set(reference_hashes.tolist())
    matched_mask = synthetic_hashes.isin(reference_set)
    matched_hashes = synthetic_hashes[matched_mask]
    distinct_matched = sorted(set(matched_hashes.tolist()))
    distinct_reference_matched = set(reference_hashes[reference_hashes.isin(distinct_matched)].tolist())
    return {
        "columns_used": usable_columns,
        "synthetic_rows": synthetic_rows,
        "reference_rows": reference_rows,
        "exact_match_count": int(matched_mask.sum()),
        "exact_match_rate": float(matched_mask.sum() / synthetic_rows),
        "distinct_matched_synthetic_rows": int(len(distinct_matched)),
        "distinct_reference_rows_matched": int(len(distinct_reference_matched)),
        "matched_row_hashes": distinct_matched[: int(max_hashes)],
        "canonical_representation": _canonical_representation_description(usable_columns),
    }


def _canonical_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    canonical = pd.DataFrame(index=frame.index)
    for column in columns:
        canonical[column] = frame[column].map(lambda value, name=column: _canonical_value(value, name))
    return canonical


def _canonical_value(value: Any, column: str) -> Any:
    if pd.isna(value):
        return "<NA>"
    if column == "Renda":
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return str(normalize_text_value(value))
    if column in {"Idade", "Dependentes", "DDD"}:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return str(normalize_text_value(value))
    normalized = normalize_text_value(value)
    if isinstance(normalized, str):
        return normalized
    if isinstance(normalized, (np.integer, int)):
        return int(normalized)
    if isinstance(normalized, (np.floating, float)):
        return round(float(normalized), 6)
    return str(normalized)


def _row_hashes(canonical: pd.DataFrame, columns: list[str]) -> pd.Series:
    def row_hash(row: pd.Series) -> str:
        payload = json.dumps([row[column] for column in columns], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return canonical.apply(row_hash, axis=1)


def _canonical_representation_description(columns: list[str]) -> dict[str, Any]:
    return {
        "columns": columns,
        "unicode_normalization": UNICODE_NORMALIZATION,
        "text_aliases": "valores textuais normalizados em NFC e aliases legados aplicados",
        "null_value": "<NA>",
        "numeric_rules": {
            "Renda": "float arredondado a 2 casas decimais",
            "Idade": "inteiro",
            "Dependentes": "inteiro",
            "DDD": "inteiro",
        },
        "derived_identifiers_excluded": ["Nome", "Data_Nascimento", "CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"],
    }


def _encode_for_distance(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    metadata: DatasetMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [column for column in metadata.model_columns if column in synthetic.columns and column in reference.columns]
    categorical = [column for column in metadata.categorical_columns(True) if column in columns]
    numeric = [column for column in metadata.numeric_columns(False) if column in columns]

    frames = []
    if numeric:
        combined_numeric = (
            pd.concat([reference[numeric], synthetic[numeric]], ignore_index=True)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        combined_numeric = combined_numeric.fillna(combined_numeric.median(numeric_only=True)).fillna(0.0)
        minimum = combined_numeric.min()
        span = (combined_numeric.max() - minimum).replace(0, 1.0)
        scaled = ((combined_numeric - minimum) / span).to_numpy(dtype=float)
        frames.append(pd.DataFrame(scaled))
    if categorical:
        combined_categorical = (
            pd.concat([reference[categorical], synthetic[categorical]], ignore_index=True)
            .astype("string")
            .fillna("<NA>")
            .astype(str)
        )
        frames.append(pd.get_dummies(combined_categorical, columns=categorical, dtype=float))
    if not frames:
        zeros = np.zeros((len(reference) + len(synthetic), 1), dtype=float)
        return zeros[len(reference) :], zeros[: len(reference)]
    combined = pd.concat(frames, axis=1).to_numpy(dtype=float)
    return combined[len(reference) :], combined[: len(reference)]


def nearest_neighbor_metrics(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    metadata: DatasetMetadata,
    max_rows: int = 1000,
) -> dict[str, Any]:
    """Calcula DCR e NNDR sobre atributos de modelo, excluindo identificadores."""
    if len(synthetic) == 0 or len(reference) == 0:
        return {"distance_to_closest_record": None, "nearest_neighbor_distance_ratio": None}
    synthetic_sample = synthetic.head(max_rows).copy()
    reference_sample = reference.head(max_rows).copy()
    synthetic_encoded, reference_encoded = _encode_for_distance(synthetic_sample, reference_sample, metadata)
    distances = np.sqrt(((synthetic_encoded[:, None, :] - reference_encoded[None, :, :]) ** 2).sum(axis=2))
    sorted_distances = np.sort(distances, axis=1)
    closest = sorted_distances[:, 0]
    ratio = None
    if sorted_distances.shape[1] > 1:
        ratio = np.divide(
            sorted_distances[:, 0],
            sorted_distances[:, 1],
            out=np.ones_like(sorted_distances[:, 0]),
            where=sorted_distances[:, 1] != 0,
        )
    return {
        "distance_to_closest_record": {
            "min": float(np.min(closest)),
            "mean": float(np.mean(closest)),
            "median": float(np.median(closest)),
        },
        "nearest_neighbor_distance_ratio": None
        if ratio is None
        else {
            "mean": float(np.mean(ratio)),
            "median": float(np.median(ratio)),
        },
        "sampled_rows": int(len(synthetic_sample)),
    }


def category_coverage(synthetic: pd.DataFrame, reference: pd.DataFrame, metadata: DatasetMetadata) -> dict[str, float]:
    """Retorna a cobertura de categorias por coluna categórica."""
    coverage: dict[str, float] = {}
    for column in metadata.categorical_columns(include_discrete_numeric=True):
        if column not in synthetic.columns or column not in reference.columns:
            continue
        ref_values = set(reference[column].astype(str).dropna().unique())
        syn_values = set(synthetic[column].astype(str).dropna().unique())
        coverage[column] = 1.0 if not ref_values else float(len(ref_values & syn_values) / len(ref_values))
    return coverage


def _partial_match_metrics(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name, columns in PARTIAL_MATCH_COLUMN_SETS.items():
        diagnostics[name] = {
            "columns_used": [column for column in columns if column in synthetic.columns],
            "train": exact_match_metrics(synthetic, train, columns, max_hashes=50),
            "holdout": exact_match_metrics(synthetic, holdout, columns, max_hashes=50),
            "diagnostic_only": True,
        }
    return diagnostics


def privacy_metrics(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
    max_nearest_neighbor_rows: int = 1000,
) -> dict[str, Any]:
    """Calcula indicadores de diversidade e risco de memorização.

    Essas métricas são apenas indicadores. Elas não provam anonimização.
    """
    metadata = metadata or default_metadata()
    columns = [
        column
        for column in metadata.model_columns
        if column in synthetic.columns and column in train.columns and column in holdout.columns
    ]
    duplicate_base = duplicate_base_row_metrics(synthetic, columns)
    exact_train = exact_match_metrics(synthetic, train, columns)
    exact_holdout = exact_match_metrics(synthetic, holdout, columns)
    coverage_train = category_coverage(synthetic, train, metadata)
    return {
        "columns_used": columns,
        "excluded_columns": metadata.proximity_excluded_columns,
        "duplicate_row_rate": duplicate_base["duplicate_row_rate"],
        "exact_train_match_rate": exact_train["exact_match_rate"],
        "exact_holdout_match_rate": exact_holdout["exact_match_rate"],
        "unique_combinations": int(duplicate_base["unique_rows"]),
        "unique_combination_rate": float(duplicate_base["unique_combination_rate"]),
        "duplicate_base_rows": duplicate_base,
        "exact_matches": {
            "train": exact_train,
            "holdout": exact_holdout,
        },
        "partial_matches": _partial_match_metrics(synthetic, train, holdout),
        "category_coverage_train": coverage_train,
        "category_coverage_holdout": category_coverage(synthetic, holdout, metadata),
        "nearest_neighbor_train": nearest_neighbor_metrics(synthetic, train, metadata, max_rows=max_nearest_neighbor_rows),
        "nearest_neighbor_holdout": nearest_neighbor_metrics(synthetic, holdout, metadata, max_rows=max_nearest_neighbor_rows),
        "mode_collapse_indicators": {
            "low_unique_combination_rate": bool(duplicate_base["unique_combination_rate"] < 0.80),
            "duplicate_row_rate": duplicate_base["duplicate_row_rate"],
            "duplicate_base_row_rate": duplicate_base["duplicate_row_rate"],
            "min_category_coverage_train": float(min(coverage_train.values() or [1.0])),
        },
        "interpretation": "Indicators only; they are not a proof of anonymization.",
    }
