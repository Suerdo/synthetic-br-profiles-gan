"""Métricas de validade e diversidade geográfica."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.domain.brazil import STATE_DDDS, STATE_MUNICIPALITIES, region_for_state
from synthetic_br_profiles_gan.domain.geography import GEOGRAPHY_CATALOG_VERSION, encode_geography_tuple, geography_key_categories


def geography_quality_report(reference: pd.DataFrame, synthetic: pd.DataFrame, geography_model_version: int = 1) -> dict[str, Any]:
    """Calcula validade relacional e cobertura geográfica."""
    validity = geography_validity_metrics(synthetic)
    diversity = geography_diversity_metrics(reference, synthetic)
    return {
        "geography_model_version": int(geography_model_version),
        "geography_catalog_version": GEOGRAPHY_CATALOG_VERSION,
        "validity": validity,
        "diversity": diversity,
        "interpretation": (
            "Métricas calculadas sobre Região, Estado, Município e DDD. O DDD é validado como permitido "
            "para o estado na tabela sintética local."
        ),
    }


def geography_validity_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Calcula validade individual e conjunta das relações geográficas."""
    total = int(len(frame))
    if total == 0:
        return _empty_validity()
    region_state = _region_state_mask(frame)
    state_municipality = _state_municipality_mask(frame)
    state_ddd = _state_ddd_mask(frame)
    known_keys = _known_key_mask(frame)
    geographic_joint = region_state & state_municipality & state_ddd & known_keys
    return {
        "known_geography_key_rate": _rate(int(known_keys.sum()), total),
        "raw_geographic_validity_rate": _rate(int(geographic_joint.sum()), total),
        "region_state_valid_rate": _rate(int(region_state.sum()), total),
        "state_municipality_valid_rate": _rate(int(state_municipality.sum()), total),
        "state_ddd_valid_rate": _rate(int(state_ddd.sum()), total),
        "geographic_joint_valid_rate": _rate(int(geographic_joint.sum()), total),
        "total_rows": total,
        "valid_rows": int(geographic_joint.sum()),
        "invalid_rows": int(total - int(geographic_joint.sum())),
        "invalid_counts": {
            "unknown_geography_key": int(total - int(known_keys.sum())),
            "region_state": int(total - int(region_state.sum())),
            "state_municipality": int(total - int(state_municipality.sum())),
            "state_ddd": int(total - int(state_ddd.sum())),
            "geographic_joint": int(total - int(geographic_joint.sum())),
        },
        "top_invalid_combinations": _top_invalid_combinations(frame, geographic_joint),
    }


def geography_diversity_metrics(reference: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Calcula cobertura e TVD de chaves e componentes geográficos."""
    ref_keys = _encoded_geo_keys(reference)
    syn_keys = _encoded_geo_keys(synthetic)
    catalog = set(geography_key_categories())
    ref_key_counts = Counter(value for value in ref_keys if value)
    syn_key_counts = Counter(value for value in syn_keys if value)
    return {
        "geography_key_coverage": _coverage(set(ref_key_counts), set(syn_key_counts)),
        "geography_key_unique_count": int(len(set(syn_key_counts))),
        "geography_key_duplicate_rate": _duplicate_rate([value for value in syn_keys if value]),
        "state_coverage": _component_coverage(reference, synthetic, "Estado"),
        "municipality_coverage": _component_coverage(reference, synthetic, "Municipio"),
        "ddd_coverage": _component_coverage(reference, synthetic, "DDD"),
        "region_coverage": _component_coverage(reference, synthetic, "Regiao"),
        "region_distribution_tvd": _column_tvd(reference, synthetic, "Regiao"),
        "state_distribution_tvd": _column_tvd(reference, synthetic, "Estado"),
        "municipality_distribution_tvd": _column_tvd(reference, synthetic, "Municipio"),
        "ddd_distribution_tvd": _column_tvd(reference, synthetic, "DDD"),
        "geography_key_distribution_tvd": _counts_tvd(ref_key_counts, syn_key_counts),
        "rare_geography_key_coverage": _rare_key_coverage(ref_key_counts, syn_key_counts),
        "catalog_key_count": int(len(catalog)),
        "reference_key_count": int(len(ref_key_counts)),
        "synthetic_key_count": int(len(syn_key_counts)),
    }


def _empty_validity() -> dict[str, Any]:
    return {
        "known_geography_key_rate": 0.0,
        "raw_geographic_validity_rate": 0.0,
        "region_state_valid_rate": 0.0,
        "state_municipality_valid_rate": 0.0,
        "state_ddd_valid_rate": 0.0,
        "geographic_joint_valid_rate": 0.0,
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "invalid_counts": {},
        "top_invalid_combinations": [],
    }


def _region_state_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"Regiao", "Estado"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    return frame.apply(lambda row: region_for_state(str(row["Estado"])) == str(row["Regiao"]), axis=1).astype(bool)


def _state_municipality_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"Estado", "Municipio"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    return frame.apply(lambda row: str(row["Municipio"]) in STATE_MUNICIPALITIES.get(str(row["Estado"]), ()), axis=1).astype(bool)


def _state_ddd_mask(frame: pd.DataFrame) -> pd.Series:
    if not {"Estado", "DDD"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)

    def valid(row: pd.Series) -> bool:
        try:
            return int(row["DDD"]) in STATE_DDDS.get(str(row["Estado"]), ())
        except (TypeError, ValueError):
            return False

    return frame.apply(valid, axis=1).astype(bool)


def _known_key_mask(frame: pd.DataFrame) -> pd.Series:
    keys = _encoded_geo_keys(frame)
    return pd.Series([value is not None for value in keys], index=frame.index)


def _encoded_geo_keys(frame: pd.DataFrame) -> list[str | None]:
    if not {"Regiao", "Estado", "Municipio", "DDD"}.issubset(frame.columns):
        return [None] * len(frame)
    return [
        encode_geography_tuple(row["Regiao"], row["Estado"], row["Municipio"], row["DDD"])
        for _, row in frame.iterrows()
    ]


def _top_invalid_combinations(frame: pd.DataFrame, valid_mask: pd.Series, limit: int = 20) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    columns = [column for column in ["Regiao", "Estado", "Municipio", "DDD"] if column in frame.columns]
    if not columns:
        return []
    invalid = frame.loc[~valid_mask, columns].copy()
    if invalid.empty:
        return []
    grouped = invalid.astype(str).value_counts().head(int(limit))
    rows: list[dict[str, Any]] = []
    for key, count in grouped.items():
        values = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, values)), "count": int(count)})
    return rows


def _component_coverage(reference: pd.DataFrame, synthetic: pd.DataFrame, column: str) -> float:
    if column not in reference.columns or column not in synthetic.columns:
        return 0.0
    reference_values = {str(value) for value in reference[column].dropna().unique()}
    synthetic_values = {str(value) for value in synthetic[column].dropna().unique()}
    return _coverage(reference_values, synthetic_values)


def _coverage(reference_values: set[str], synthetic_values: set[str]) -> float:
    if not reference_values:
        return 1.0
    return float(len(reference_values & synthetic_values) / len(reference_values))


def _column_tvd(reference: pd.DataFrame, synthetic: pd.DataFrame, column: str) -> float | None:
    if column not in reference.columns or column not in synthetic.columns:
        return None
    ref_counts = Counter(reference[column].astype(str).dropna().tolist())
    syn_counts = Counter(synthetic[column].astype(str).dropna().tolist())
    return _counts_tvd(ref_counts, syn_counts)


def _counts_tvd(reference_counts: Counter[str], synthetic_counts: Counter[str]) -> float:
    reference_total = sum(reference_counts.values())
    synthetic_total = sum(synthetic_counts.values())
    if reference_total <= 0 or synthetic_total <= 0:
        return 0.0
    keys = set(reference_counts) | set(synthetic_counts)
    return float(
        0.5
        * sum(
            abs(reference_counts.get(key, 0) / reference_total - synthetic_counts.get(key, 0) / synthetic_total)
            for key in keys
        )
    )


def _duplicate_rate(values: list[str]) -> float:
    if not values:
        return 0.0
    return float((len(values) - len(set(values))) / len(values))


def _rare_key_coverage(reference_counts: Counter[str], synthetic_counts: Counter[str], threshold: float = 0.01) -> dict[str, Any]:
    total = sum(reference_counts.values())
    rare = {
        key: count
        for key, count in reference_counts.items()
        if total > 0 and count > 0 and (count / total) < float(threshold)
    }
    reproduced = [key for key in rare if synthetic_counts.get(key, 0) > 0]
    return {
        "threshold": float(threshold),
        "rare_key_count": int(len(rare)),
        "reproduced_rare_key_count": int(len(reproduced)),
        "coverage": 1.0 if not rare else float(len(reproduced) / len(rare)),
        "missing_rare_keys": sorted(set(rare) - set(reproduced))[:100],
    }


def _rate(valid: int, total: int) -> float:
    return 0.0 if int(total) <= 0 else float(int(valid) / int(total))
