"""Métricas de qualidade estatística para dados tabulares sintéticos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from synthetic_br_profiles_gan.evaluation.privacy import privacy_metrics
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric[np.isfinite(numeric)].dropna()


def numeric_column_metrics(reference: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """Calcula métricas detalhadas para uma coluna numérica."""
    ref = _finite_numeric(reference)
    syn = _finite_numeric(synthetic)
    if ref.empty or syn.empty:
        return {"error": "empty_numeric_series"}
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    ref_mean = float(ref.mean())
    syn_mean = float(syn.mean())
    absolute_mean_diff = abs(syn_mean - ref_mean)
    relative_mean_diff = None if ref_mean == 0 else absolute_mean_diff / abs(ref_mean)
    ks = stats.ks_2samp(ref.to_numpy(), syn.to_numpy())
    wasserstein = float(stats.wasserstein_distance(ref.to_numpy(), syn.to_numpy()))
    iqr = float(ref.quantile(0.75) - ref.quantile(0.25))
    ref_std = float(ref.std(ddof=1)) if len(ref) > 1 else 0.0
    normalization_scale = iqr if iqr > 0 else ref_std
    normalized_wasserstein = None if normalization_scale <= 0 else wasserstein / normalization_scale
    return {
        "reference": {
            "mean": ref_mean,
            "median": float(ref.median()),
            "std": _safe_float(ref.std()),
            "min": float(ref.min()),
            "max": float(ref.max()),
            "quantiles": {str(q): float(ref.quantile(q)) for q in quantiles},
        },
        "synthetic": {
            "mean": syn_mean,
            "median": float(syn.median()),
            "std": _safe_float(syn.std()),
            "min": float(syn.min()),
            "max": float(syn.max()),
            "quantiles": {str(q): float(syn.quantile(q)) for q in quantiles},
        },
        "absolute_mean_diff": float(absolute_mean_diff),
        "relative_mean_diff": _safe_float(relative_mean_diff),
        "wasserstein_distance": wasserstein,
        "wasserstein_distance_normalized": _safe_float(normalized_wasserstein),
        "wasserstein_normalization": "reference_iqr_fallback_std",
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "median_diff": float(abs(float(syn.median()) - float(ref.median()))),
        "std_diff": float(abs(float(syn.std()) - float(ref.std()))),
    }


def categorical_column_metrics(reference: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    """Calcula métricas de distribuição categórica para uma coluna."""
    ref = reference.astype("string").fillna("<NA>").astype(str)
    syn = synthetic.astype("string").fillna("<NA>").astype(str)
    categories = sorted(set(ref.unique()).union(set(syn.unique())))
    ref_freq = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    syn_freq = syn.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    diff = (syn_freq - ref_freq).abs()
    return {
        "reference_frequency": {category: float(ref_freq[category]) for category in categories},
        "synthetic_frequency": {category: float(syn_freq[category]) for category in categories},
        "proportion_diff": {category: float(diff[category]) for category in categories},
        "missing_categories": sorted(set(ref.unique()) - set(syn.unique())),
        "unexpected_categories": sorted(set(syn.unique()) - set(ref.unique())),
        "total_variation_distance": float(0.5 * diff.sum()),
    }


def correlation_metrics(reference: pd.DataFrame, synthetic: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    """Compara matrizes de correlação de Pearson e Spearman."""
    available = [column for column in numeric_columns if column in reference.columns and column in synthetic.columns]
    if len(available) < 2:
        return {"pearson": {}, "spearman": {}, "summary": {"max_abs_difference": 0.0, "mean_abs_difference": 0.0}}
    ref_numeric = reference[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    syn_numeric = synthetic[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result: dict[str, Any] = {}
    all_diffs: list[float] = []
    for method in ("pearson", "spearman"):
        ref_corr = ref_numeric.corr(method=method).fillna(0.0)
        syn_corr = syn_numeric.corr(method=method).fillna(0.0)
        diff = (syn_corr - ref_corr).abs()
        all_diffs.extend(diff.to_numpy().reshape(-1).tolist())
        result[method] = {
            "reference": ref_corr.to_dict(),
            "synthetic": syn_corr.to_dict(),
            "absolute_difference": diff.to_dict(),
        }
    result["summary"] = {
        "max_abs_difference": float(max(all_diffs) if all_diffs else 0.0),
        "mean_abs_difference": float(np.mean(all_diffs) if all_diffs else 0.0),
    }
    return result


def categorical_relationship_metrics(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Compara distribuições em crosstab para relações categóricas relevantes."""
    relationships: dict[str, Any] = {}
    for left, right in pairs:
        if left not in reference.columns or right not in reference.columns:
            continue
        if left not in synthetic.columns or right not in synthetic.columns:
            continue
        ref_tab = pd.crosstab(reference[left], reference[right], normalize="all")
        syn_tab = pd.crosstab(synthetic[left], synthetic[right], normalize="all")
        ref_aligned, syn_aligned = ref_tab.align(syn_tab, join="outer", axis=None, fill_value=0.0)
        diff = (syn_aligned - ref_aligned).abs()
        relationships[f"{left}__{right}"] = {
            "total_variation_distance": float(0.5 * diff.to_numpy().sum()),
            "max_cell_difference": float(diff.to_numpy().max() if diff.size else 0.0),
        }
    return relationships


def grouped_income_metrics(reference: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Compara resumos de renda entre colunas importantes de agrupamento."""
    groups = ["Regiao", "Escolaridade", "Ocupacao"]
    metrics: dict[str, Any] = {}
    for group in groups:
        if group not in reference.columns or group not in synthetic.columns or "Renda" not in reference.columns:
            continue
        ref = reference.groupby(group)["Renda"].mean()
        syn = synthetic.groupby(group)["Renda"].mean()
        ref_aligned, syn_aligned = ref.align(syn, join="outer")
        diff = (syn_aligned - ref_aligned).abs()
        metrics[group] = {
            "mean_income_reference": {str(key): _safe_float(value) for key, value in ref_aligned.items()},
            "mean_income_synthetic": {str(key): _safe_float(value) for key, value in syn_aligned.items()},
            "absolute_difference": {str(key): _safe_float(value) for key, value in diff.items()},
        }

    if {"Idade", "Renda"}.issubset(reference.columns) and {"Idade", "Renda"}.issubset(synthetic.columns):
        bins = [17, 24, 34, 44, 54, 64, 85]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        ref_age = reference.assign(Faixa_Etaria=pd.cut(reference["Idade"], bins=bins, labels=labels))
        syn_age = synthetic.assign(Faixa_Etaria=pd.cut(synthetic["Idade"], bins=bins, labels=labels))
        ref_mean = ref_age.groupby("Faixa_Etaria", observed=False)["Renda"].mean()
        syn_mean = syn_age.groupby("Faixa_Etaria", observed=False)["Renda"].mean()
        diff = (syn_mean - ref_mean).abs()
        metrics["Faixa_Etaria"] = {
            "mean_income_reference": {str(key): _safe_float(value) for key, value in ref_mean.items()},
            "mean_income_synthetic": {str(key): _safe_float(value) for key, value in syn_mean.items()},
            "absolute_difference": {str(key): _safe_float(value) for key, value in diff.items()},
        }
    return metrics


def evaluate_against_reference(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
) -> dict[str, Any]:
    """Avalia dados sintéticos contra um split de referência."""
    metadata = metadata or default_metadata()
    numeric_columns = [column for column in metadata.numeric_columns() if column in reference.columns and column in synthetic.columns]
    categorical_columns = [
        column for column in metadata.categorical_columns(include_discrete_numeric=True) if column in reference.columns and column in synthetic.columns
    ]
    return {
        "numeric": {
            column: numeric_column_metrics(reference[column], synthetic[column])
            for column in numeric_columns
        },
        "categorical": {
            column: categorical_column_metrics(reference[column], synthetic[column])
            for column in categorical_columns
        },
        "correlations": correlation_metrics(reference, synthetic, numeric_columns),
        "categorical_relationships": categorical_relationship_metrics(
            reference,
            synthetic,
            [
                ("Regiao", "Estado"),
                ("Estado", "Municipio"),
                ("Estado", "DDD"),
                ("Escolaridade", "Ocupacao"),
                ("Idade", "Estado_Civil"),
            ],
        ),
        "grouped_income": grouped_income_metrics(reference, synthetic),
    }


def evaluate_synthetic_data(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
    max_nearest_neighbor_rows: int = 1000,
) -> dict[str, Any]:
    """Compara dados sintéticos com os splits de treino e holdout."""
    metadata = metadata or default_metadata()
    synthetic_model = synthetic[[column for column in metadata.model_columns if column in synthetic.columns]].copy()
    return {
        "against_train": evaluate_against_reference(train, synthetic_model, metadata),
        "against_holdout": evaluate_against_reference(holdout, synthetic_model, metadata),
        "privacy": privacy_metrics(synthetic_model, train, holdout, metadata, max_nearest_neighbor_rows=max_nearest_neighbor_rows),
        "row_counts": {
            "synthetic": int(len(synthetic_model)),
            "train": int(len(train)),
            "holdout": int(len(holdout)),
        },
    }
