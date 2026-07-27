"""Diversity and privacy-risk indicators for synthetic data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata


def duplicate_row_rate(df: pd.DataFrame) -> float:
    """Return the rate of duplicated rows."""
    if len(df) == 0:
        return 0.0
    return float(df.duplicated().sum() / len(df))


def exact_match_rate(synthetic: pd.DataFrame, reference: pd.DataFrame, columns: list[str]) -> float:
    """Return exact row match rate against reference over selected columns."""
    usable_columns = [column for column in columns if column in synthetic.columns and column in reference.columns]
    if len(synthetic) == 0 or not usable_columns:
        return 0.0
    ref_records = {tuple(row) for row in reference[usable_columns].astype(str).itertuples(index=False, name=None)}
    matches = sum(tuple(row) in ref_records for row in synthetic[usable_columns].astype(str).itertuples(index=False, name=None))
    return float(matches / len(synthetic))


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
    """Compute DCR and NNDR over model attributes, excluding identifiers."""
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
    """Return category coverage per categorical column."""
    coverage: dict[str, float] = {}
    for column in metadata.categorical_columns(include_discrete_numeric=True):
        if column not in synthetic.columns or column not in reference.columns:
            continue
        ref_values = set(reference[column].astype(str).dropna().unique())
        syn_values = set(synthetic[column].astype(str).dropna().unique())
        coverage[column] = 1.0 if not ref_values else float(len(ref_values & syn_values) / len(ref_values))
    return coverage


def privacy_metrics(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
    max_nearest_neighbor_rows: int = 1000,
) -> dict[str, Any]:
    """Compute diversity and memorization-risk indicators.

    These metrics are indicators only. They do not prove anonymization.
    """
    metadata = metadata or default_metadata()
    columns = [
        column
        for column in metadata.model_columns
        if column in synthetic.columns and column in train.columns and column in holdout.columns
    ]
    unique_ratio = 0.0 if len(synthetic) == 0 else synthetic[columns].drop_duplicates().shape[0] / len(synthetic)
    return {
        "columns_used": columns,
        "excluded_columns": metadata.proximity_excluded_columns,
        "duplicate_row_rate": duplicate_row_rate(synthetic[columns]),
        "exact_train_match_rate": exact_match_rate(synthetic, train, columns),
        "exact_holdout_match_rate": exact_match_rate(synthetic, holdout, columns),
        "unique_combinations": int(synthetic[columns].drop_duplicates().shape[0]),
        "unique_combination_rate": float(unique_ratio),
        "category_coverage_train": category_coverage(synthetic, train, metadata),
        "category_coverage_holdout": category_coverage(synthetic, holdout, metadata),
        "nearest_neighbor_train": nearest_neighbor_metrics(synthetic, train, metadata, max_rows=max_nearest_neighbor_rows),
        "nearest_neighbor_holdout": nearest_neighbor_metrics(synthetic, holdout, metadata, max_rows=max_nearest_neighbor_rows),
        "mode_collapse_indicators": {
            "low_unique_combination_rate": bool(unique_ratio < 0.80),
            "duplicate_row_rate": duplicate_row_rate(synthetic[columns]),
            "min_category_coverage_train": float(min(category_coverage(synthetic, train, metadata).values() or [1.0])),
        },
        "interpretation": "Indicators only; they are not a proof of anonymization.",
    }
