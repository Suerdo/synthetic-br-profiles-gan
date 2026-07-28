"""Métricas específicas para o vocabulário categórico em português brasileiro."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.domain.occupations import (
    get_occupation_profile,
    is_occupation_compatible,
)
from synthetic_br_profiles_gan.localization import (
    CATEGORICAL_VOCABULARY_VERSION,
    DATA_LOCALE,
    LEGACY_CATEGORY_ALIASES,
    UNICODE_NORMALIZATION,
)
from synthetic_br_profiles_gan.metadata import MODEL_COLUMNS, DatasetMetadata, default_metadata


DEFAULT_RARE_OCCUPATION_THRESHOLD = 0.01
DEFAULT_MINIMUM_INCOME_GROUP_COUNT = 20
DEFAULT_LOW_COUNT_THRESHOLD = 10
INCOME_COMPARISON_PAIRS = (
    ("Médico", "Atendente"),
    ("Engenheiro", "Serviços Gerais"),
    ("Gerente", "Auxiliar Administrativo"),
    ("Analista de Dados", "Operador de Caixa"),
)


def evaluate_vocabulary_v2_quality(
    reference: pd.DataFrame,
    raw: pd.DataFrame,
    final: pd.DataFrame,
    metadata: DatasetMetadata | None = None,
    requested_rows: int | None = None,
    validation_report: dict[str, Any] | None = None,
    rare_threshold: float = DEFAULT_RARE_OCCUPATION_THRESHOLD,
    minimum_income_group_count: int = DEFAULT_MINIMUM_INCOME_GROUP_COUNT,
    low_count_threshold: int = DEFAULT_LOW_COUNT_THRESHOLD,
) -> dict[str, Any]:
    """Avalia cobertura, coerência e renda condicionada no vocabulário categórico 2."""
    metadata = metadata or default_metadata()
    occupations = tuple(str(value) for value in metadata.columns["Ocupacao"].categories or [])
    requested = int(requested_rows if requested_rows is not None else len(final))
    reference_counts = _category_counts(reference, "Ocupacao", occupations)
    raw_counts = _category_counts(raw, "Ocupacao", occupations)
    final_counts = _category_counts(final, "Ocupacao", occupations)

    occupation_metrics = {
        "canonical_occupation_count": len(occupations),
        "occupation_reference_count": reference_counts,
        "occupation_raw_count": raw_counts,
        "occupation_final_count": final_counts,
        "occupation_reference_coverage": _coverage(reference_counts),
        "occupation_raw_coverage": _coverage(raw_counts),
        "occupation_final_coverage": _coverage(final_counts),
        "missing_occupations_raw": [occupation for occupation, count in raw_counts.items() if count == 0],
        "missing_occupations_final": [occupation for occupation, count in final_counts.items() if count == 0],
        "unexpected_occupations_raw": _unexpected_values(raw, "Ocupacao", occupations),
        "unexpected_occupations_final": _unexpected_values(final, "Ocupacao", occupations),
        "legacy_occupations_raw": _legacy_values(raw, "Ocupacao"),
        "legacy_occupations_final": _legacy_values(final, "Ocupacao"),
        "occupation_distribution_distance_raw": _total_variation_distance(reference_counts, raw_counts),
        "occupation_distribution_distance_final": _total_variation_distance(reference_counts, final_counts),
        "occupation_distribution_distance": _total_variation_distance(reference_counts, final_counts),
        "occupation_entropy_reference": _entropy(reference_counts),
        "occupation_entropy_raw": _entropy(raw_counts),
        "occupation_entropy_final": _entropy(final_counts),
        "most_frequent_occupation_share_raw": _most_frequent_share(raw_counts),
        "most_frequent_occupation_share_final": _most_frequent_share(final_counts),
    }

    rare = _rare_occupation_metrics(
        reference_counts=reference_counts,
        raw_counts=raw_counts,
        final_counts=final_counts,
        threshold=float(rare_threshold),
    )
    coherence = _coherence_metrics(raw, final)
    income = _income_metrics(
        reference=reference,
        raw=raw,
        final=final,
        occupations=occupations,
        minimum_income_group_count=int(minimum_income_group_count),
    )
    diversity = _diversity_metrics(reference, raw, final, low_count_threshold=int(low_count_threshold))
    locale = _locale_metrics(reference, raw, final, metadata)
    gates = _vocabulary_gates(
        final=final,
        metadata=metadata,
        requested_rows=requested,
        validation_report=validation_report or {},
        occupation_metrics=occupation_metrics,
        coherence=coherence,
        locale=locale,
    )
    return {
        "schema_version": 1,
        "data_locale": DATA_LOCALE,
        "unicode_normalization": UNICODE_NORMALIZATION,
        "categorical_vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "requested_rows": requested,
        "rare_occupation_threshold": float(rare_threshold),
        "minimum_income_group_count": int(minimum_income_group_count),
        "low_count_threshold": int(low_count_threshold),
        "occupation": occupation_metrics,
        "rare_occupations": rare,
        "coherence": coherence,
        "income_by_occupation": income["by_occupation"],
        "income_comparisons": income["comparisons"],
        "gender_audit": income["gender_audit"],
        "diversity": diversity,
        "locale": locale,
        "quality_gates": gates,
        "methodological_notice": (
            "Gênero não é utilizado como parâmetro no cálculo sintético da renda. "
            "Eventuais diferenças amostrais não representam uma regra implementada."
        ),
    }


def _category_counts(df: pd.DataFrame, column: str, categories: tuple[str, ...]) -> dict[str, int]:
    if column not in df.columns:
        return {category: 0 for category in categories}
    counts = df[column].astype(str).value_counts(dropna=False).to_dict()
    return {category: int(counts.get(category, 0)) for category in categories}


def _coverage(counts: dict[str, int]) -> float:
    return float(sum(1 for count in counts.values() if int(count) > 0) / len(counts)) if counts else 0.0


def _shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: float(int(value) / total) for key, value in counts.items()}


def _total_variation_distance(reference_counts: dict[str, int], observed_counts: dict[str, int]) -> float:
    reference = _shares(reference_counts)
    observed = _shares(observed_counts)
    categories = sorted(set(reference) | set(observed))
    return float(0.5 * sum(abs(reference.get(category, 0.0) - observed.get(category, 0.0)) for category in categories))


def _entropy(counts: dict[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if int(count) <= 0:
            continue
        probability = int(count) / total
        entropy -= probability * math.log(probability)
    return float(entropy)


def _most_frequent_share(counts: dict[str, int]) -> float:
    total = sum(int(value) for value in counts.values())
    return float(max(counts.values()) / total) if total > 0 and counts else 0.0


def _unexpected_values(df: pd.DataFrame, column: str, categories: tuple[str, ...]) -> list[str]:
    if column not in df.columns:
        return []
    valid = set(categories)
    values = sorted({str(value) for value in df[column].dropna().unique() if str(value) not in valid})
    return values


def _legacy_values(df: pd.DataFrame, column: str | None = None) -> list[str]:
    if column is not None:
        if column not in df.columns:
            return []
        values = df[column].dropna().astype(str)
    else:
        text_columns = [name for name in df.columns if pd.api.types.is_object_dtype(df[name]) or pd.api.types.is_string_dtype(df[name])]
        values = pd.Series([str(value) for name in text_columns for value in df[name].dropna().tolist()])
    legacy = sorted({value for value in values if value in LEGACY_CATEGORY_ALIASES})
    return legacy


def _rare_occupation_metrics(
    reference_counts: dict[str, int],
    raw_counts: dict[str, int],
    final_counts: dict[str, int],
    threshold: float,
) -> dict[str, Any]:
    reference_total = max(sum(reference_counts.values()), 1)
    rare_rows: dict[str, dict[str, Any]] = {}
    for occupation, count in reference_counts.items():
        share = float(count / reference_total)
        if count > 0 and share < threshold:
            rare_rows[occupation] = {
                "reference_count": int(count),
                "reference_share": share,
                "raw_count": int(raw_counts.get(occupation, 0)),
                "raw_share": _shares(raw_counts).get(occupation, 0.0),
                "final_count": int(final_counts.get(occupation, 0)),
                "final_share": _shares(final_counts).get(occupation, 0.0),
                "reproduced_raw": bool(raw_counts.get(occupation, 0) > 0),
                "reproduced_final": bool(final_counts.get(occupation, 0) > 0),
            }
    return {
        "criterion": f"participação inferior a {threshold:.2%} no holdout",
        "occupations": rare_rows,
        "rare_occupation_count": len(rare_rows),
        "reproduced_raw_count": sum(1 for row in rare_rows.values() if row["reproduced_raw"]),
        "reproduced_final_count": sum(1 for row in rare_rows.values() if row["reproduced_final"]),
        "missing_rare_occupations_raw": [name for name, row in rare_rows.items() if not row["reproduced_raw"]],
        "missing_rare_occupations_final": [name for name, row in rare_rows.items() if not row["reproduced_final"]],
    }


def _coherence_metrics(raw: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    raw_education = _invalid_education_occupation(raw)
    final_education = _invalid_education_occupation(final)
    raw_age = _invalid_age_occupation(raw)
    final_age = _invalid_age_occupation(final)
    return {
        "education_occupation_valid_rate_raw": _valid_rate(len(raw), raw_education),
        "education_occupation_valid_rate_final": _valid_rate(len(final), final_education),
        "education_occupation_invalid_count_raw": int(sum(item["count"] for item in raw_education)),
        "education_occupation_invalid_count_final": int(sum(item["count"] for item in final_education)),
        "invalid_education_occupation_raw": raw_education,
        "invalid_education_occupation_final": final_education,
        "age_occupation_valid_rate_raw": _valid_rate(len(raw), raw_age),
        "age_occupation_valid_rate_final": _valid_rate(len(final), final_age),
        "age_occupation_invalid_count_raw": int(sum(item["count"] for item in raw_age)),
        "age_occupation_invalid_count_final": int(sum(item["count"] for item in final_age)),
        "invalid_age_occupation_raw": raw_age,
        "invalid_age_occupation_final": final_age,
        "age_rule_interpretation": "Somente restrições obrigatórias do catálogo são bloqueantes; tendências probabilísticas não são invalidadas.",
    }


def _valid_rate(total_rows: int, invalid_rows: list[dict[str, Any]]) -> float:
    if total_rows <= 0:
        return 0.0
    invalid_count = sum(int(item["count"]) for item in invalid_rows)
    return float(max(total_rows - invalid_count, 0) / total_rows)


def _invalid_education_occupation(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"Escolaridade", "Ocupacao"}
    if not required.issubset(df.columns):
        return []
    counts: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        occupation = str(row["Ocupacao"])
        education = str(row["Escolaridade"])
        profile = get_occupation_profile(occupation)
        if profile is None or education not in profile.allowed_education:
            counts[(education, occupation)] += 1
    return [
        {"Escolaridade": education, "Ocupacao": occupation, "count": int(count)}
        for (education, occupation), count in counts.most_common(20)
    ]


def _invalid_age_occupation(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"Idade", "Ocupacao"}
    if not required.issubset(df.columns):
        return []
    counts: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        occupation = str(row["Ocupacao"])
        profile = get_occupation_profile(occupation)
        try:
            age = int(row["Idade"])
        except (TypeError, ValueError):
            counts[(str(row["Idade"]), occupation)] += 1
            continue
        if profile is None or age < profile.minimum_age or (profile.maximum_age is not None and age > profile.maximum_age):
            counts[(str(age), occupation)] += 1
    return [
        {"Idade": age, "Ocupacao": occupation, "count": int(count)}
        for (age, occupation), count in counts.most_common(20)
    ]


def _income_metrics(
    reference: pd.DataFrame,
    raw: pd.DataFrame,
    final: pd.DataFrame,
    occupations: tuple[str, ...],
    minimum_income_group_count: int,
) -> dict[str, Any]:
    frames = {"reference": reference, "raw": raw, "final": final}
    by_occupation = {
        stage: _income_summary_by_occupation(frame, occupations, minimum_income_group_count)
        for stage, frame in frames.items()
    }
    comparisons: dict[str, Any] = {}
    for left, right in INCOME_COMPARISON_PAIRS:
        comparison_name = f"{left} versus {right}"
        comparisons[comparison_name] = {
            stage: _income_pair_comparison(summaries, left, right)
            for stage, summaries in by_occupation.items()
        }
    return {
        "by_occupation": by_occupation,
        "comparisons": comparisons,
        "gender_audit": {stage: _income_by_gender(frame) for stage, frame in frames.items()},
    }


def _income_summary_by_occupation(
    df: pd.DataFrame,
    occupations: tuple[str, ...],
    minimum_income_group_count: int,
) -> dict[str, dict[str, Any]]:
    if not {"Ocupacao", "Renda"}.issubset(df.columns):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for occupation in occupations:
        values = pd.to_numeric(df.loc[df["Ocupacao"].astype(str) == occupation, "Renda"], errors="coerce").dropna()
        if values.empty:
            result[occupation] = {"count": 0, "sufficient_sample": False}
            continue
        result[occupation] = {
            "count": int(len(values)),
            "sufficient_sample": bool(len(values) >= minimum_income_group_count),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _income_pair_comparison(summaries: dict[str, dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_summary = summaries.get(left, {})
    right_summary = summaries.get(right, {})
    if not left_summary.get("count") or not right_summary.get("count"):
        return {"available": False}
    left_mean = left_summary.get("mean")
    right_mean = right_summary.get("mean")
    overlap = bool(left_summary.get("min") <= right_summary.get("max") and right_summary.get("min") <= left_summary.get("max"))
    return {
        "available": True,
        "left_count": left_summary.get("count"),
        "right_count": right_summary.get("count"),
        "left_mean": left_mean,
        "right_mean": right_mean,
        "mean_difference": None if left_mean is None or right_mean is None else float(left_mean - right_mean),
        "distributions_overlap": overlap,
    }


def _income_by_gender(df: pd.DataFrame) -> dict[str, Any]:
    if not {"Genero", "Renda"}.issubset(df.columns):
        return {"available": False}
    rows: dict[str, Any] = {}
    for gender, group in df.groupby(df["Genero"].astype(str), dropna=False):
        values = pd.to_numeric(group["Renda"], errors="coerce").dropna()
        rows[str(gender)] = {
            "count": int(len(values)),
            "mean": None if values.empty else float(values.mean()),
            "median": None if values.empty else float(values.median()),
        }
    return {
        "available": True,
        "by_gender": rows,
        "notice": "Gênero não é utilizado como parâmetro no cálculo sintético da renda.",
    }


def _diversity_metrics(reference: pd.DataFrame, raw: pd.DataFrame, final: pd.DataFrame, low_count_threshold: int) -> dict[str, Any]:
    frames = {"reference": reference, "raw": raw, "final": final}
    return {
        stage: {
            "distinct_base_combinations": _distinct_base_combinations(frame),
            "duplicate_base_row_rate": _duplicate_base_row_rate(frame),
            "low_count_occupation_count": _low_count_categories(frame, "Ocupacao", low_count_threshold),
            "education_occupation_coverage": _pair_coverage(frame, "Escolaridade", "Ocupacao"),
            "region_occupation_coverage": _pair_coverage(frame, "Regiao", "Ocupacao"),
        }
        for stage, frame in frames.items()
    }


def _base_columns_present(df: pd.DataFrame) -> list[str]:
    return [column for column in MODEL_COLUMNS if column in df.columns]


def _distinct_base_combinations(df: pd.DataFrame) -> int:
    columns = _base_columns_present(df)
    return int(df[columns].drop_duplicates().shape[0]) if columns else 0


def _duplicate_base_row_rate(df: pd.DataFrame) -> float:
    columns = _base_columns_present(df)
    if not columns or len(df) == 0:
        return 0.0
    return float(df[columns].duplicated().sum() / len(df))


def _low_count_categories(df: pd.DataFrame, column: str, threshold: int) -> int:
    if column not in df.columns:
        return 0
    counts = df[column].astype(str).value_counts()
    return int((counts < int(threshold)).sum())


def _pair_coverage(df: pd.DataFrame, left: str, right: str) -> int:
    if not {left, right}.issubset(df.columns):
        return 0
    return int(df[[left, right]].drop_duplicates().shape[0])


def _locale_metrics(reference: pd.DataFrame, raw: pd.DataFrame, final: pd.DataFrame, metadata: DatasetMetadata) -> dict[str, Any]:
    frames = {"reference": reference, "raw": raw, "final": final}
    categorical_columns = [
        name
        for name, column in metadata.columns.items()
        if column.kind == "categorical" and name in set(reference.columns) | set(raw.columns) | set(final.columns)
    ]
    result: dict[str, Any] = {}
    for stage, frame in frames.items():
        result[f"unicode_nfc_valid_{stage}"] = _unicode_nfc_valid(frame)
        result[f"legacy_value_count_{stage}"] = len(_legacy_values(frame))
        result[f"legacy_values_{stage}"] = _legacy_values(frame)
        result[f"unexpected_categorical_values_{stage}"] = _unexpected_categorical_values(frame, metadata, categorical_columns)
    result["data_locale"] = DATA_LOCALE
    result["unicode_normalization"] = UNICODE_NORMALIZATION
    result["output_vocabulary_version"] = CATEGORICAL_VOCABULARY_VERSION
    return result


def _unicode_nfc_valid(df: pd.DataFrame) -> bool:
    for column in df.columns:
        if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
            continue
        for value in df[column].dropna().astype(str):
            if unicodedata.normalize("NFC", value) != value:
                return False
    return True


def _unexpected_categorical_values(
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    unexpected: dict[str, list[str]] = {}
    for column in categorical_columns:
        if column not in df.columns:
            continue
        categories = tuple(str(value) for value in metadata.columns[column].categories or [])
        values = _unexpected_values(df, column, categories)
        if values:
            unexpected[column] = values
    return unexpected


def _vocabulary_gates(
    final: pd.DataFrame,
    metadata: DatasetMetadata,
    requested_rows: int,
    validation_report: dict[str, Any],
    occupation_metrics: dict[str, Any],
    coherence: dict[str, Any],
    locale: dict[str, Any],
) -> dict[str, Any]:
    income = pd.to_numeric(final["Renda"], errors="coerce") if "Renda" in final.columns else pd.Series(dtype=float)
    income_meta = metadata.columns["Renda"]
    blocking_checks = [
        {
            "name": "final_rows_exact",
            "mandatory": True,
            "passed": bool(len(final) == int(requested_rows)),
            "value": int(len(final)),
            "expected": int(requested_rows),
            "interpretation": "Quantidade final de linhas exportáveis após validação.",
        },
        {
            "name": "schema_final_valid",
            "mandatory": True,
            "passed": bool(validation_report.get("is_valid", False)),
            "value": bool(validation_report.get("is_valid", False)),
            "interpretation": "Validação estrutural final do schema completo.",
        },
        {
            "name": "final_categories_canonical",
            "mandatory": True,
            "passed": bool(
                not occupation_metrics["unexpected_occupations_final"]
                and not locale.get("unexpected_categorical_values_final")
            ),
            "value": {
                "unexpected_occupations_final": occupation_metrics["unexpected_occupations_final"],
                "unexpected_categorical_values_final": locale.get("unexpected_categorical_values_final", {}),
            },
            "interpretation": "Categorias finais devem pertencer ao vocabulário canônico.",
        },
        {
            "name": "final_structural_compatibility",
            "mandatory": True,
            "passed": bool(
                coherence["education_occupation_invalid_count_final"] == 0
                and coherence["age_occupation_invalid_count_final"] == 0
            ),
            "value": {
                "education_occupation_invalid_count_final": coherence["education_occupation_invalid_count_final"],
                "age_occupation_invalid_count_final": coherence["age_occupation_invalid_count_final"],
            },
            "interpretation": "Ocupação final deve ser estruturalmente compatível com escolaridade e idade.",
        },
        {
            "name": "no_legacy_categories_final",
            "mandatory": True,
            "passed": bool(locale.get("legacy_value_count_final", 0) == 0),
            "value": locale.get("legacy_values_final", []),
            "interpretation": "A saída final não deve manter valores legados sem acentuação.",
        },
        {
            "name": "income_within_limits_final",
            "mandatory": True,
            "passed": bool(
                not income.empty
                and income.notna().all()
                and income.min() >= float(income_meta.min_value)
                and income.max() <= float(income_meta.max_value)
            ),
            "value": {
                "min": None if income.empty else float(income.min()),
                "max": None if income.empty else float(income.max()),
            },
            "interpretation": "Renda final deve permanecer dentro dos limites configurados.",
        },
        {
            "name": "unicode_nfc_final",
            "mandatory": True,
            "passed": bool(locale.get("unicode_nfc_valid_final", False)),
            "value": locale.get("unicode_nfc_valid_final", False),
            "interpretation": "Textos finais devem estar normalizados em Unicode NFC.",
        },
    ]
    diagnostic_checks = [
        {
            "name": "occupation_final_coverage",
            "mandatory": False,
            "value": occupation_metrics["occupation_final_coverage"],
            "interpretation": "Cobertura das 37 ocupações; métrica diagnóstica nesta primeira avaliação.",
        },
        {
            "name": "occupation_distribution_distance_final",
            "mandatory": False,
            "value": occupation_metrics["occupation_distribution_distance_final"],
            "interpretation": "Distância de variação total da distribuição de ocupações; sem limite bloqueante nesta fase.",
        },
        {
            "name": "occupation_entropy_final",
            "mandatory": False,
            "value": occupation_metrics["occupation_entropy_final"],
            "interpretation": "Entropia da distribuição de ocupações; usada para diagnosticar concentração.",
        },
    ]
    failures = [check for check in blocking_checks if check["mandatory"] and not check["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_checks": blocking_checks,
        "diagnostic_checks": diagnostic_checks,
        "blocking_failures": failures,
    }
