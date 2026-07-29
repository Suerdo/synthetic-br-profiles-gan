"""Métricas de realismo condicional da renda sintética."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONAL_INCOME_GROUPINGS: dict[str, list[str]] = {
    "ocupacao": ["Ocupacao"],
    "ocupacao_escolaridade": ["Ocupacao", "Escolaridade"],
    "ocupacao_faixa_etaria": ["Ocupacao", "Faixa_Etaria"],
    "ocupacao_regiao": ["Ocupacao", "Regiao"],
    "ocupacao_escolaridade_faixa_etaria": ["Ocupacao", "Escolaridade", "Faixa_Etaria"],
}


def age_band(age: Any) -> str:
    """Retorna a faixa etária sintética usada nas métricas condicionais."""
    try:
        value = int(age)
    except (TypeError, ValueError):
        return "Não avaliado"
    if value < 25:
        return "18-24"
    if value < 35:
        return "25-34"
    if value < 45:
        return "35-44"
    if value < 60:
        return "45-59"
    return "60+"


def conditional_income_report(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    minimum_group_rows: int = 30,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compara distribuições de renda em grupos condicionais."""
    prepared = {
        "reference": _prepare_income_frame(reference),
        "final": _prepare_income_frame(synthetic),
    }
    if raw is not None:
        prepared["raw"] = _prepare_income_frame(raw)

    summary_rows: list[dict[str, Any]] = []
    for stage, frame in prepared.items():
        summary_rows.extend(
            conditional_income_summary_rows(
                frame,
                stage=stage,
                minimum_group_rows=minimum_group_rows,
                model=model,
                seed=seed,
            )
        )

    comparison_rows = conditional_income_comparison_rows(
        prepared["reference"],
        prepared["final"],
        stage="final",
        minimum_group_rows=minimum_group_rows,
        model=model,
        seed=seed,
    )
    if raw is not None:
        comparison_rows.extend(
            conditional_income_comparison_rows(
                prepared["reference"],
                prepared["raw"],
                stage="raw",
                minimum_group_rows=minimum_group_rows,
                model=model,
                seed=seed,
            )
        )

    tail_rows = conditional_income_tail_events(
        prepared["reference"],
        prepared["final"],
        stage="final",
        minimum_group_rows=minimum_group_rows,
        model=model,
        seed=seed,
    )
    if raw is not None:
        tail_rows.extend(
            conditional_income_tail_events(
                prepared["reference"],
                prepared["raw"],
                stage="raw",
                minimum_group_rows=minimum_group_rows,
                model=model,
                seed=seed,
            )
        )

    summary = _income_plausibility_summary(summary_rows, comparison_rows, tail_rows)
    return {
        "minimum_group_rows": int(minimum_group_rows),
        "groupings": CONDITIONAL_INCOME_GROUPINGS,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "tail_events": tail_rows,
        "summary": summary,
        "interpretation": (
            "Métricas diagnósticas de realismo condicional. Diferenças em caudas e percentis "
            "não são, isoladamente, prova de erro ou de representatividade populacional."
        ),
    }


def conditional_income_summary_rows(
    frame: pd.DataFrame,
    *,
    stage: str,
    minimum_group_rows: int = 30,
    model: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Resume renda por ocupação, escolaridade, faixa etária e região."""
    if frame.empty or "Renda" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for grouping_name, columns in CONDITIONAL_INCOME_GROUPINGS.items():
        usable_columns = [column for column in columns if column in frame.columns]
        if len(usable_columns) != len(columns):
            continue
        for values, group in frame.groupby(usable_columns, dropna=False):
            income = _finite_income(group["Renda"])
            if len(income) < int(minimum_group_rows):
                continue
            values_tuple = values if isinstance(values, tuple) else (values,)
            quantiles = income.quantile([0.05, 0.25, 0.75, 0.90, 0.95, 0.99])
            p25 = float(quantiles.loc[0.25])
            p75 = float(quantiles.loc[0.75])
            row = {
                "model": model,
                "seed": seed,
                "stage": stage,
                "grouping": grouping_name,
                "group_columns": "|".join(usable_columns),
                "group_key": _group_key(usable_columns, values_tuple),
                "count": int(len(income)),
                "mean": float(income.mean()),
                "median": float(income.median()),
                "std": _safe_float(income.std(ddof=1)),
                "p05": float(quantiles.loc[0.05]),
                "p25": p25,
                "p75": p75,
                "p90": float(quantiles.loc[0.90]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "min": float(income.min()),
                "max": float(income.max()),
                "interquartile_range": float(p75 - p25),
                "upper_tail_threshold": float(quantiles.loc[0.95]),
                "lower_tail_threshold": float(quantiles.loc[0.05]),
                "upper_tail_rate": float((income > quantiles.loc[0.95]).mean()),
                "lower_tail_rate": float((income < quantiles.loc[0.05]).mean()),
            }
            for column, value in zip(usable_columns, values_tuple):
                row[column] = value
            rows.append(row)
    return rows


def conditional_income_comparison_rows(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    stage: str,
    minimum_group_rows: int = 30,
    model: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Compara referência e sintético em cada grupo condicional suficiente."""
    if reference.empty or synthetic.empty:
        return []
    rows: list[dict[str, Any]] = []
    for grouping_name, columns in CONDITIONAL_INCOME_GROUPINGS.items():
        usable_columns = [column for column in columns if column in reference.columns and column in synthetic.columns]
        if len(usable_columns) != len(columns):
            continue
        reference_groups = {
            _normalize_group_values(values): group
            for values, group in reference.groupby(usable_columns, dropna=False)
            if len(_finite_income(group["Renda"])) >= int(minimum_group_rows)
        }
        synthetic_groups = {
            _normalize_group_values(values): group
            for values, group in synthetic.groupby(usable_columns, dropna=False)
            if len(_finite_income(group["Renda"])) >= int(minimum_group_rows)
        }
        for values, ref_group in reference_groups.items():
            syn_group = synthetic_groups.get(values)
            if syn_group is None:
                continue
            ref_income = _finite_income(ref_group["Renda"])
            syn_income = _finite_income(syn_group["Renda"])
            if len(ref_income) < int(minimum_group_rows) or len(syn_income) < int(minimum_group_rows):
                continue
            ref_quantiles = ref_income.quantile([0.95, 0.99])
            syn_quantiles = syn_income.quantile([0.95, 0.99])
            row = {
                "model": model,
                "seed": seed,
                "stage": stage,
                "grouping": grouping_name,
                "group_columns": "|".join(usable_columns),
                "group_key": _group_key(usable_columns, values),
                "reference_count": int(len(ref_income)),
                "synthetic_count": int(len(syn_income)),
                "conditional_income_wasserstein": float(stats.wasserstein_distance(ref_income, syn_income)),
                "conditional_income_median_difference": float(syn_income.median() - ref_income.median()),
                "conditional_income_p95_difference": float(syn_quantiles.loc[0.95] - ref_quantiles.loc[0.95]),
                "conditional_income_p99_difference": float(syn_quantiles.loc[0.99] - ref_quantiles.loc[0.99]),
                "high_tail_excess_rate": float((syn_income > ref_quantiles.loc[0.95]).mean() - 0.05),
                "reference_p95": float(ref_quantiles.loc[0.95]),
                "synthetic_p95": float(syn_quantiles.loc[0.95]),
                "reference_p99": float(ref_quantiles.loc[0.99]),
                "synthetic_p99": float(syn_quantiles.loc[0.99]),
            }
            for column, value in zip(usable_columns, values):
                row[column] = value
            rows.append(row)
    rows.extend(_rank_consistency_rows(reference, synthetic, stage=stage, model=model, seed=seed))
    return rows


def conditional_income_tail_events(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    stage: str,
    minimum_group_rows: int = 30,
    model: str | None = None,
    seed: int | None = None,
    max_events: int = 500,
) -> list[dict[str, Any]]:
    """Lista eventos de cauda superior sem identificadores derivados."""
    required = ["Ocupacao", "Escolaridade", "Faixa_Etaria", "Regiao", "Renda"]
    if any(column not in reference.columns or column not in synthetic.columns for column in required):
        return []
    rows: list[dict[str, Any]] = []
    grouping = ["Ocupacao", "Escolaridade", "Faixa_Etaria", "Regiao"]
    thresholds: dict[tuple[Any, ...], float] = {}
    for values, group in reference.groupby(grouping, dropna=False):
        income = _finite_income(group["Renda"])
        if len(income) >= int(minimum_group_rows):
            thresholds[_normalize_group_values(values)] = float(income.quantile(0.99))
    for values, group in synthetic.groupby(grouping, dropna=False):
        key = _normalize_group_values(values)
        threshold = thresholds.get(key)
        if threshold is None:
            continue
        events = group[pd.to_numeric(group["Renda"], errors="coerce") > threshold].head(max_events - len(rows))
        for _, event in events.iterrows():
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "stage": stage,
                    "Ocupacao": event.get("Ocupacao"),
                    "Escolaridade": event.get("Escolaridade"),
                    "Faixa_Etaria": event.get("Faixa_Etaria"),
                    "Regiao": event.get("Regiao"),
                    "Renda": float(event.get("Renda")),
                    "percentil_condicional": "acima_p99_referencia",
                    "limiar_condicional": threshold,
                }
            )
            if len(rows) >= max_events:
                return rows
    return rows


def _prepare_income_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "Idade" in prepared.columns and "Faixa_Etaria" not in prepared.columns:
        prepared["Faixa_Etaria"] = prepared["Idade"].map(age_band)
    if "Renda" in prepared.columns:
        prepared["Renda"] = pd.to_numeric(prepared["Renda"], errors="coerce").round(2)
    return prepared


def _finite_income(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values[np.isfinite(values)].dropna()


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _normalize_group_values(values: Any) -> tuple[Any, ...]:
    return values if isinstance(values, tuple) else (values,)


def _group_key(columns: list[str], values: tuple[Any, ...]) -> str:
    return "|".join(f"{column}={value}" for column, value in zip(columns, values))


def _rank_consistency_rows(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    stage: str,
    model: str | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    if "Ocupacao" not in reference.columns or "Ocupacao" not in synthetic.columns:
        return []
    ref_medians = reference.groupby("Ocupacao")["Renda"].median(numeric_only=False)
    syn_medians = synthetic.groupby("Ocupacao")["Renda"].median(numeric_only=False)
    common = sorted(set(ref_medians.index) & set(syn_medians.index))
    if len(common) < 2:
        correlation = None
    else:
        correlation = stats.spearmanr([ref_medians.loc[item] for item in common], [syn_medians.loc[item] for item in common]).correlation
    return [
        {
            "model": model,
            "seed": seed,
            "stage": stage,
            "grouping": "ocupacao_rank",
            "group_columns": "Ocupacao",
            "group_key": "todas_ocupacoes_comuns",
            "reference_count": int(reference["Ocupacao"].nunique()),
            "synthetic_count": int(synthetic["Ocupacao"].nunique()),
            "conditional_group_coverage": float(len(common) / max(int(reference["Ocupacao"].nunique()), 1)),
            "occupation_income_rank_correlation": None if correlation is None or np.isnan(correlation) else float(correlation),
            "occupation_education_income_consistency": None,
        }
    ]


def _income_plausibility_summary(
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    tail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons = [row for row in comparison_rows if row.get("grouping") != "ocupacao_rank"]
    if not comparisons:
        return {
            "status": "Não avaliado",
            "conditional_groups_compared": 0,
            "tail_events": int(len(tail_rows)),
        }
    p95_diffs = [abs(float(row["conditional_income_p95_difference"])) for row in comparisons]
    p99_diffs = [abs(float(row["conditional_income_p99_difference"])) for row in comparisons]
    wasserstein = [float(row["conditional_income_wasserstein"]) for row in comparisons]
    rank_rows = [row for row in comparison_rows if row.get("grouping") == "ocupacao_rank"]
    rank_correlation = rank_rows[0].get("occupation_income_rank_correlation") if rank_rows else None
    return {
        "status": "diagnóstico",
        "summary_groups": int(len(summary_rows)),
        "conditional_groups_compared": int(len(comparisons)),
        "max_conditional_income_wasserstein": float(max(wasserstein)),
        "mean_conditional_income_wasserstein": float(np.mean(wasserstein)),
        "max_abs_p95_difference": float(max(p95_diffs)),
        "max_abs_p99_difference": float(max(p99_diffs)),
        "occupation_income_rank_correlation": rank_correlation,
        "tail_events": int(len(tail_rows)),
        "groups_with_excessive_tail": int(
            sum(1 for row in comparisons if float(row.get("high_tail_excess_rate", 0.0)) > 0.10)
        ),
    }
