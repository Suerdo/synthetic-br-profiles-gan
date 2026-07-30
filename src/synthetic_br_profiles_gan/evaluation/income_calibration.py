"""Avaliação controlada de versões do modelo sintético de renda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from synthetic_br_profiles_gan.calibration import DEFAULT_CALIBRATION_CONFIG, _sample_income
from synthetic_br_profiles_gan.domain.occupations import get_occupation_profile

REQUIRED_INCOME_OCCUPATIONS = (
    "Mecânico",
    "Eletricista",
    "Pedreiro",
    "Motorista",
    "Técnico de Informática",
    "Técnico de Enfermagem",
    "Serviços Gerais",
    "Atendente",
    "Vendedor",
    "Professor",
    "Engenheiro",
    "Médico",
    "Autônomo",
    "Microempreendedor",
)

INCOME_VERSION_SPECS = (
    {"name": "income_v1", "version": 1, "variant": "historical", "classification": "historical_baseline"},
    {"name": "income_v2", "version": 2, "variant": "current", "classification": "current_baseline"},
    {"name": "income_v3_candidate_a", "version": 3, "variant": "candidate_a", "classification": "candidate"},
    {"name": "income_v3_candidate_b", "version": 3, "variant": "selected", "classification": "selected_calibration"},
)


@dataclass(frozen=True)
class IncomeVersionSpec:
    """Especificação de uma versão candidata do modelo sintético de renda."""

    name: str
    version: int
    variant: str
    classification: str


def default_income_version_specs() -> tuple[IncomeVersionSpec, ...]:
    """Retorna as versões comparadas no refinamento de renda v3."""
    return tuple(IncomeVersionSpec(**spec) for spec in INCOME_VERSION_SPECS)


def distribution_overlap_coefficient(left: np.ndarray, right: np.ndarray, bins: int = 80) -> float:
    """Calcula a sobreposição aproximada entre duas distribuições contínuas."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0:
        return 0.0
    lower = float(min(left.min(), right.min()))
    upper = float(max(left.max(), right.max()))
    if lower == upper:
        return 1.0
    edges = np.linspace(lower, upper, int(bins) + 1)
    left_hist, _ = np.histogram(left, bins=edges, density=True)
    right_hist, _ = np.histogram(right, bins=edges, density=True)
    return float(np.sum(np.minimum(left_hist, right_hist) * np.diff(edges)))


def quantile_overlap(left: np.ndarray, right: np.ndarray, lower_q: float = 0.25, upper_q: float = 0.75) -> float:
    """Calcula a sobreposição entre intervalos interquantis."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or right.size == 0:
        return 0.0
    left_low, left_high = np.quantile(left, [lower_q, upper_q])
    right_low, right_high = np.quantile(right, [lower_q, upper_q])
    intersection = max(min(left_high, right_high) - max(left_low, right_low), 0.0)
    union = max(max(left_high, right_high) - min(left_low, right_low), 1e-9)
    return float(intersection / union)


def _sample_for_spec(
    spec: IncomeVersionSpec,
    occupation: str,
    seed: int,
    rows: int,
) -> np.ndarray:
    profile = get_occupation_profile(occupation)
    if profile is None:
        raise ValueError(f"Unknown occupation: {occupation}")
    rng = np.random.default_rng(int(seed) + int(spec.version) * 1009 + sum(ord(char) for char in spec.name))
    regions = ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul")
    income_config = DEFAULT_CALIBRATION_CONFIG["income"]
    values: list[float] = []
    for _ in range(int(rows)):
        education = str(rng.choice(profile.allowed_education))
        maximum_age = int(profile.maximum_age or 75)
        age = int(rng.integers(int(profile.minimum_age), maximum_age + 1))
        region = str(rng.choice(regions))
        values.append(
            _sample_income(
                rng,
                age,
                education,
                occupation,
                region,
                float(income_config["min"]),
                float(income_config["max"]),
                int(spec.version),
                str(spec.variant),
            )
        )
    return np.asarray(values, dtype=float)


def _summary(values: np.ndarray, threshold: float) -> dict[str, Any]:
    quantiles = np.quantile(values, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    mean = float(values.mean()) if len(values) else 0.0
    return {
        "count": int(len(values)),
        "mean": mean,
        "median": float(quantiles[1]),
        "std": std,
        "coefficient_of_variation": float(0.0 if mean == 0 else std / abs(mean)),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "min": float(values.min()),
        "max": float(values.max()),
        "interquartile_range": float(quantiles[2] - quantiles[0]),
        "high_tail_rate": float((values > threshold).mean()),
    }


def run_income_calibration_analysis(
    seeds: list[int],
    rows_per_occupation: int,
    occupations: tuple[str, ...] = REQUIRED_INCOME_OCCUPATIONS,
    specs: tuple[IncomeVersionSpec, ...] | None = None,
) -> dict[str, Any]:
    """Compara versões de renda com amostragem condicional controlada."""
    specs = specs or default_income_version_specs()
    samples: dict[tuple[str, int, str], np.ndarray] = {}
    for spec in specs:
        for seed in seeds:
            for occupation in occupations:
                samples[(spec.name, int(seed), occupation)] = _sample_for_spec(spec, occupation, int(seed), int(rows_per_occupation))

    thresholds: dict[tuple[int, str], float] = {}
    for seed in seeds:
        for occupation in occupations:
            combined = np.concatenate([samples[(spec.name, int(seed), occupation)] for spec in specs])
            thresholds[(int(seed), occupation)] = float(np.quantile(combined, 0.95))

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            for occupation in occupations:
                values = samples[(spec.name, int(seed), occupation)]
                rows.append(
                    {
                        "version_name": spec.name,
                        "income_model_version": int(spec.version),
                        "income_model_variant": spec.variant,
                        "classification": spec.classification,
                        "seed": int(seed),
                        "Ocupacao": occupation,
                        **_summary(values, thresholds[(int(seed), occupation)]),
                    }
                )

    summary = pd.DataFrame(rows)
    compression_rows = _compression_rows(summary)
    overlap_rows = _overlap_rows(samples, specs, seeds, occupations)
    ranking = _rank_metrics(summary)
    selected = _select_income_calibration(summary, compression_rows, overlap_rows)
    return {
        "rows": rows,
        "compression": compression_rows,
        "overlap": overlap_rows,
        "ranking": ranking,
        "selected_calibration": selected,
        "threshold_definition": "p95 combinado das versões comparadas por seed e ocupação; não é teto rígido.",
        "interpretation": (
            "As métricas comparam parâmetros sintéticos de renda. Elas não representam estatísticas oficiais "
            "do mercado de trabalho brasileiro."
        ),
    }


def _compression_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_v1 = summary[summary["version_name"].eq("income_v1")]
    base_v2 = summary[summary["version_name"].eq("income_v2")]
    for _, row in summary.iterrows():
        if row["version_name"] == "income_v1":
            continue
        match_v1 = base_v1[(base_v1["seed"].eq(row["seed"])) & (base_v1["Ocupacao"].eq(row["Ocupacao"]))]
        match_v2 = base_v2[(base_v2["seed"].eq(row["seed"])) & (base_v2["Ocupacao"].eq(row["Ocupacao"]))]
        if match_v1.empty:
            continue
        ref1 = match_v1.iloc[0]
        ref2 = match_v2.iloc[0] if not match_v2.empty else None
        item = {
            "version_name": row["version_name"],
            "seed": int(row["seed"]),
            "Ocupacao": row["Ocupacao"],
        }
        for metric in ["median", "p95", "p99", "std", "interquartile_range", "high_tail_rate"]:
            item[f"{metric}_change_vs_v1"] = _relative_change(row[metric], ref1[metric])
            if ref2 is not None and row["version_name"] != "income_v2":
                item[f"{metric}_change_vs_v2"] = _relative_change(row[metric], ref2[metric])
        rows.append(item)
    return rows


def _relative_change(value: float, reference: float) -> float | None:
    reference = float(reference)
    if abs(reference) < 1e-9:
        return None
    return float((float(value) - reference) / reference)


def _overlap_rows(
    samples: dict[tuple[str, int, str], np.ndarray],
    specs: tuple[IncomeVersionSpec, ...],
    seeds: list[int],
    occupations: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_names = ("income_v1", "income_v2")
    for spec in specs:
        for baseline in baseline_names:
            if spec.name == baseline:
                continue
            for seed in seeds:
                for occupation in occupations:
                    current = samples[(spec.name, int(seed), occupation)]
                    reference = samples[(baseline, int(seed), occupation)]
                    rows.append(
                        {
                            "version_name": spec.name,
                            "baseline": baseline,
                            "seed": int(seed),
                            "Ocupacao": occupation,
                            "distribution_overlap_coefficient": distribution_overlap_coefficient(current, reference),
                            "quantile_overlap": quantile_overlap(current, reference),
                            "wasserstein": float(stats.wasserstein_distance(current, reference)),
                        }
                    )
    return rows


def _rank_metrics(summary: pd.DataFrame) -> list[dict[str, Any]]:
    expected_order = summary[summary["version_name"].eq("income_v1")].groupby("Ocupacao")["median"].mean().sort_values()
    expected_rank = {occupation: rank for rank, occupation in enumerate(expected_order.index, start=1)}
    rows: list[dict[str, Any]] = []
    for version_name, group in summary.groupby("version_name"):
        medians = group.groupby("Ocupacao")["median"].mean().sort_values()
        current_rank = {occupation: rank for rank, occupation in enumerate(medians.index, start=1)}
        aligned = [occupation for occupation in expected_order.index if occupation in current_rank]
        if len(aligned) < 2:
            correlation = None
            inversions = 0
        else:
            correlation = float(stats.spearmanr([expected_rank[item] for item in aligned], [current_rank[item] for item in aligned]).statistic)
            inversions = sum(
                1
                for left_index, left in enumerate(aligned)
                for right in aligned[left_index + 1 :]
                if (expected_rank[left] - expected_rank[right]) * (current_rank[left] - current_rank[right]) < 0
            )
        rows.append(
            {
                "version_name": version_name,
                "median_rank_correlation": correlation,
                "aggregate_rank_inversions": int(inversions),
                "dispersion_ratio": float(group["std"].mean() / max(summary[summary["version_name"].eq("income_v1")]["std"].mean(), 1e-9)),
            }
        )
    return rows


def _select_income_calibration(summary: pd.DataFrame, compression_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = ["income_v3_candidate_a", "income_v3_candidate_b"]
    scores: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_compression = [row for row in compression_rows if row["version_name"] == candidate]
        candidate_overlap = [row for row in overlap_rows if row["version_name"] == candidate]
        p95_shrinkage = np.mean([abs(row.get("p95_change_vs_v1") or 0.0) for row in candidate_compression])
        p99_shrinkage = np.mean([abs(row.get("p99_change_vs_v1") or 0.0) for row in candidate_compression])
        overlap_v2 = np.mean(
            [row["distribution_overlap_coefficient"] for row in candidate_overlap if row["baseline"] == "income_v2"]
            or [0.0]
        )
        tail_rate = float(summary[summary["version_name"].eq(candidate)]["high_tail_rate"].mean())
        # Escore diagnóstico: aproxima a cauda intermediária e preserva sobreposição.
        score = float((1.0 - abs(p95_shrinkage - 0.28)) + (1.0 - abs(p99_shrinkage - 0.30)) + overlap_v2 - abs(tail_rate - 0.04))
        scores.append({"version_name": candidate, "score": score, "mean_high_tail_rate": tail_rate, "mean_overlap_vs_v2": float(overlap_v2)})
    selected = max(scores, key=lambda item: item["score"])
    return {
        "version_name": selected["version_name"],
        "income_model_version": 3,
        "income_model_variant": "selected" if selected["version_name"] == "income_v3_candidate_b" else "candidate_a",
        "classification": "selected_calibration",
        "selection_scores": scores,
    }
