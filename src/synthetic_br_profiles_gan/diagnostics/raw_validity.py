"""Diagnóstico da validade bruta e dependência do pós-processamento."""

from __future__ import annotations

import hashlib
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.config import load_yaml_config
from synthetic_br_profiles_gan.domain.brazil import STATE_DDDS, STATE_MUNICIPALITIES, region_for_state
from synthetic_br_profiles_gan.domain.occupations import get_occupation_profile
from synthetic_br_profiles_gan.evaluation.metrics import numeric_column_metrics
from synthetic_br_profiles_gan.generation import select_valid_candidates
from synthetic_br_profiles_gan.generators.demographics import (
    criar_estado_identificadores,
    criar_faker,
    finalizar_perfis_sinteticos,
)
from synthetic_br_profiles_gan.manifest import write_json
from synthetic_br_profiles_gan.metadata import MODEL_COLUMNS, DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


DIAGNOSTIC_MODEL_COLUMNS = tuple(MODEL_COLUMNS)
TRANSITION_COLUMNS = (
    "Regiao",
    "Estado",
    "Municipio",
    "DDD",
    "Escolaridade",
    "Ocupacao",
    "Idade",
    "Estado_Civil",
    "Dependentes",
    "Renda",
)

RULE_LABELS = {
    "known_categories": "Categorias conhecidas",
    "age_domain": "Idade",
    "income_domain": "Renda",
    "dependents_domain": "Dependentes",
    "region_state": "Região × Estado",
    "state_municipality": "Estado × Município",
    "state_ddd": "Estado × DDD",
    "occupation_education": "Ocupação × Escolaridade",
    "occupation_age": "Ocupação × Idade",
    "marital_age": "Estado Civil × Idade",
    "geographic_joint": "Validade conjunta de localização",
    "professional_joint": "Validade conjunta profissional",
    "non_relational_joint": "Validade não relacional",
    "structural_global": "Validade estrutural global",
}

RULE_COLUMNS = {
    "known_categories": DIAGNOSTIC_MODEL_COLUMNS,
    "age_domain": ("Idade",),
    "income_domain": ("Renda",),
    "dependents_domain": ("Dependentes",),
    "region_state": ("Regiao", "Estado"),
    "state_municipality": ("Estado", "Municipio"),
    "state_ddd": ("Estado", "DDD"),
    "occupation_education": ("Ocupacao", "Escolaridade"),
    "occupation_age": ("Ocupacao", "Idade"),
    "marital_age": ("Estado_Civil", "Idade"),
    "geographic_joint": ("Regiao", "Estado", "Municipio", "DDD"),
    "professional_joint": ("Idade", "Escolaridade", "Ocupacao", "Renda"),
    "non_relational_joint": DIAGNOSTIC_MODEL_COLUMNS,
    "structural_global": DIAGNOSTIC_MODEL_COLUMNS,
}


@dataclass(frozen=True)
class DiagnosticCandidates:
    """Candidatos brutos e finais de uma geração diagnóstica."""

    raw_candidates: pd.DataFrame
    final_candidates: pd.DataFrame
    selected_indices: list[int]
    global_valid_mask: pd.Series
    accounting: dict[str, Any]
    candidate_validation: dict[str, Any]


def metric_semantics() -> dict[str, Any]:
    """Retorna semântica formal das taxas usadas no diagnóstico raw/final."""
    return {
        "raw_structural_validity_rate": {
            "numerator": "linhas brutas selecionadas que passam na validação estrutural de colunas-base",
            "denominator": "linhas selecionadas para o dataset final",
            "unit": "taxa entre 0 e 1",
            "stage": "raw",
            "population": "raw_selected",
            "notes": "A linha é avaliada antes de aliases finais, reparos geográficos, clipping numérico e seleção global.",
        },
        "final_structural_validity_rate": {
            "numerator": "linhas finais selecionadas que passam na validação estrutural do schema final",
            "denominator": "linhas selecionadas para o dataset final",
            "unit": "taxa entre 0 e 1",
            "stage": "final",
            "population": "final_selected",
            "notes": "Na confirmação analisada, o denominador é 20.000 por seed.",
        },
        "postprocessing_repair_rate": {
            "numerator": "diferença positiva entre a taxa final e a taxa bruta de validade estrutural",
            "denominator": "linhas selecionadas para o dataset final",
            "unit": "taxa entre 0 e 1",
            "stage": "raw_final_comparison",
            "population": "raw_selected e final_selected",
            "notes": (
                "Este campo histórico não conta linhas com campos modificados. Ele mede ganho líquido de validade "
                "entre raw e final. Reparo e substituição por campo são reportados separadamente."
            ),
        },
        "postprocessing_rejection_rate": {
            "numerator": "candidatos rejeitados pelas regras globais de validação",
            "denominator": "todos os candidatos pós-processados gerados em todos os batches",
            "unit": "taxa entre 0 e 1",
            "stage": "generation",
            "population": "final_candidates",
            "notes": "Não inclui excedentes válidos que não foram selecionados apenas porque o alvo de linhas já foi atingido.",
        },
        "candidate_acceptance_rate": {
            "numerator": "candidatos aceitos pelas validações locais de batch",
            "denominator": "todos os candidatos pós-processados gerados em todos os batches",
            "unit": "taxa entre 0 e 1",
            "stage": "generation",
            "population": "final_candidates",
            "notes": "Equivale a batch_acceptance_rate quando disponível no accounting.",
        },
        "global_acceptance_rate": {
            "numerator": "candidatos aceitos pela validação global após concatenar batches",
            "denominator": "todos os candidatos pós-processados gerados em todos os batches",
            "unit": "taxa entre 0 e 1",
            "stage": "generation",
            "population": "final_candidates",
            "notes": "Captura restrições globais, como duplicidade entre batches.",
        },
    }


def diagnose_rule_validity(raw: pd.DataFrame, metadata: DatasetMetadata | None = None) -> dict[str, Any]:
    """Calcula validade bruta por regra e interseções de falhas."""
    metadata = metadata or default_metadata()
    frame = raw.reset_index(drop=True)
    masks = raw_rule_masks(frame, metadata)
    rows = [_rule_summary(frame, rule_id, mask) for rule_id, mask in masks.items()]
    failure_frame = pd.DataFrame({rule_id: ~mask.astype(bool) for rule_id, mask in masks.items()})
    intersections = failure_intersections(failure_frame)
    return {
        "rows": rows,
        "masks": masks,
        "intersections": intersections,
    }


def raw_rule_masks(raw: pd.DataFrame, metadata: DatasetMetadata | None = None) -> dict[str, pd.Series]:
    """Retorna uma máscara de validade para cada regra diagnóstica."""
    metadata = metadata or default_metadata()
    frame = raw.reset_index(drop=True)
    index = frame.index
    known = _known_category_mask(frame, metadata)
    age = _numeric_domain_mask(frame, "Idade", 18, 85, integer=True)
    income = _numeric_domain_mask(frame, "Renda", 800.0, 50000.0, integer=False)
    dependents = _numeric_domain_mask(frame, "Dependentes", 0, 6, integer=True)
    region_state = _region_state_mask(frame)
    state_municipality = _state_municipality_mask(frame)
    state_ddd = _state_ddd_mask(frame)
    occupation_education = _occupation_education_mask(frame)
    occupation_age = _occupation_age_mask(frame)
    marital_age = _marital_age_mask(frame)
    category_subset = _known_category_mask(frame, metadata, columns=("Regiao", "Estado", "Municipio", "DDD"))
    professional_categories = _known_category_mask(frame, metadata, columns=("Escolaridade", "Ocupacao"))
    non_relational = known & age & income & dependents
    structural = validate_profile_dataframe(frame, metadata=metadata, final=False).valid_mask.reset_index(drop=True).astype(bool)
    return {
        "known_categories": known,
        "age_domain": age,
        "income_domain": income,
        "dependents_domain": dependents,
        "region_state": region_state,
        "state_municipality": state_municipality,
        "state_ddd": state_ddd,
        "occupation_education": occupation_education,
        "occupation_age": occupation_age,
        "marital_age": marital_age,
        "geographic_joint": category_subset & region_state & state_municipality & state_ddd,
        "professional_joint": professional_categories & age & income & occupation_education & occupation_age,
        "non_relational_joint": non_relational,
        "structural_global": structural.reindex(index).fillna(False).astype(bool),
    }


def failure_intersections(failure_frame: pd.DataFrame) -> dict[str, Any]:
    """Resume linhas com múltiplas falhas e coocorrência entre regras."""
    if failure_frame.empty:
        return {
            "failure_count_distribution": {"one_failure": 0, "two_failures": 0, "three_or_more_failures": 0},
            "frequent_rule_combinations": [],
            "cooccurrence_rows": [],
        }
    counts = failure_frame.sum(axis=1).astype(int)
    combination_counts: Counter[tuple[str, ...]] = Counter()
    for _, row in failure_frame.iterrows():
        failed = tuple(rule for rule, value in row.items() if bool(value))
        if failed:
            combination_counts[failed] += 1
    cooccurrence_rows: list[dict[str, Any]] = []
    rules = list(failure_frame.columns)
    denominator = int(len(failure_frame))
    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1 :]:
            count = int((failure_frame[left] & failure_frame[right]).sum())
            if count:
                cooccurrence_rows.append(
                    {
                        "rule_a": left,
                        "rule_b": right,
                        "count": count,
                        "rate": _rate(count, denominator),
                    }
                )
    return {
        "failure_count_distribution": {
            "one_failure": int((counts == 1).sum()),
            "two_failures": int((counts == 2).sum()),
            "three_or_more_failures": int((counts >= 3).sum()),
        },
        "frequent_rule_combinations": [
            {
                "rules": list(rules),
                "count": int(count),
                "rate": _rate(int(count), denominator),
            }
            for rules, count in combination_counts.most_common(20)
        ],
        "cooccurrence_rows": cooccurrence_rows,
    }


def postprocessing_field_changes(
    raw_candidates: pd.DataFrame,
    final_candidates: pd.DataFrame,
    selected_indices: list[int],
    global_valid_mask: pd.Series | None = None,
) -> dict[str, Any]:
    """Compara colunas-base brutas e finais sem expor identificadores derivados."""
    raw = raw_candidates.reset_index(drop=True)
    final = final_candidates.reset_index(drop=True)
    selected_set = set(int(index) for index in selected_indices)
    selected_mask = pd.Series([index in selected_set for index in raw.index], index=raw.index)
    if global_valid_mask is None:
        global_valid = selected_mask.copy()
    else:
        global_valid = global_valid_mask.reset_index(drop=True).reindex(raw.index).fillna(False).astype(bool)
    changed_by_field: dict[str, pd.Series] = {}
    transition_rows: list[dict[str, Any]] = []
    field_summaries: list[dict[str, Any]] = []
    for column in TRANSITION_COLUMNS:
        before = _canonical_compare_series(raw[column], column)
        after = _canonical_compare_series(final[column], column)
        changed = before.ne(after)
        changed_by_field[column] = changed
        field_summaries.append(
            {
                "field": column,
                "total": int(len(raw)),
                "modified_rows": int(changed.sum()),
                "modified_rate": _rate(int(changed.sum()), int(len(raw))),
                "selected_modified_rows": int((changed & selected_mask).sum()),
                "rejected_modified_rows": int((changed & ~selected_mask).sum()),
            }
        )
        transitions = pd.DataFrame({"before": before[changed], "after": after[changed], "selected": selected_mask[changed]})
        if not transitions.empty:
            grouped = transitions.groupby(["before", "after"], dropna=False)
            for (before_value, after_value), group in grouped.size().sort_values(ascending=False).head(20).items():
                transition_mask = changed & before.eq(before_value) & after.eq(after_value)
                transition_rows.append(
                    {
                        "field": column,
                        "before": str(before_value),
                        "after": str(after_value),
                        "count": int(group),
                        "selected_count": int((transition_mask & selected_mask).sum()),
                        "rejected_count": int((transition_mask & ~selected_mask).sum()),
                    }
                )
    any_change = pd.concat(changed_by_field.values(), axis=1).any(axis=1) if changed_by_field else pd.Series(False, index=raw.index)
    change_count = pd.concat(changed_by_field.values(), axis=1).sum(axis=1).astype(int) if changed_by_field else pd.Series(0, index=raw.index)
    meaningful_replacement_fields = {"Idade", "Estado", "Escolaridade", "Ocupacao", "Estado_Civil", "Dependentes"}
    replacement_mask = pd.Series(False, index=raw.index)
    for column in meaningful_replacement_fields:
        if column in changed_by_field:
            replacement_mask |= changed_by_field[column]
    repair_mask = any_change & ~replacement_mask
    selected_valid = selected_mask
    invalid_rejected = ~global_valid
    surplus_valid = global_valid & ~selected_mask
    not_selected = ~selected_mask
    return {
        "field_summaries": field_summaries,
        "transition_rows": transition_rows,
        "classification": {
            "total_candidates": int(len(raw)),
            "repaired_rows": int((repair_mask & selected_valid).sum()),
            "replaced_rows": int((replacement_mask & selected_valid).sum()),
            "rejected_rows": int(invalid_rejected.sum()),
            "unchanged_rows": int((~any_change & selected_valid).sum()),
            "multiple_field_change_rows": int((change_count.gt(1) & selected_valid).sum()),
            "selected_rows": int(selected_valid.sum()),
            "surplus_valid_rows": int(surplus_valid.sum()),
            "not_selected_rows": int(not_selected.sum()),
        },
        "notes": (
            "Reparo indica ajustes de campos derivados da relação geográfica ou arredondamento/clipping sem substituir "
            "a ocupação, escolaridade, estado civil, estado, idade ou dependentes. Substituição indica mudança em campos "
            "semânticos centrais. Rejeição indica candidato inválido pelas regras globais; excedentes válidos "
            "não selecionados são contabilizados separadamente."
        ),
    }


def wasserstein_income_diagnostic(
    holdout: pd.DataFrame,
    raw_selected: pd.DataFrame,
    final_selected: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula distância de renda absoluta em BRL e normalizada."""
    raw_metrics = numeric_column_metrics(holdout["Renda"], raw_selected["Renda"])
    final_metrics = numeric_column_metrics(holdout["Renda"], final_selected["Renda"])
    holdout_income = pd.to_numeric(holdout["Renda"], errors="coerce").dropna().astype(float)
    iqr = float(holdout_income.quantile(0.75) - holdout_income.quantile(0.25))
    std = float(holdout_income.std(ddof=1)) if len(holdout_income) > 1 else 0.0
    scale = iqr if iqr > 0 else std
    return {
        "wasserstein_distance_absolute_brl": {
            "raw": raw_metrics.get("wasserstein_distance"),
            "final": final_metrics.get("wasserstein_distance"),
            "unit": "BRL",
        },
        "wasserstein_distance_normalized": {
            "raw": raw_metrics.get("wasserstein_distance_normalized"),
            "final": final_metrics.get("wasserstein_distance_normalized"),
            "unit": "reference_iqr_fallback_std",
        },
        "normalization_scale": {
            "value": scale,
            "unit": "BRL",
            "method": "IQR do holdout; fallback para desvio-padrão se IQR <= 0",
        },
    }


def missing_occupation_diagnostic(
    occupation: str,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    final_candidates: pd.DataFrame,
    selected_indices: list[int],
) -> dict[str, Any]:
    """Resume frequência de uma ocupação ausente no resultado selecionado."""
    selected = set(int(index) for index in selected_indices)
    selected_mask = pd.Series([index in selected for index in raw_candidates.reset_index(drop=True).index])
    raw = raw_candidates.reset_index(drop=True)
    final = final_candidates.reset_index(drop=True)
    return {
        "occupation": occupation,
        "train": _value_frequency(train["Ocupacao"], occupation),
        "holdout": _value_frequency(holdout["Ocupacao"], occupation),
        "raw_selected": _value_frequency(raw.loc[selected_mask, "Ocupacao"], occupation),
        "final_selected": _value_frequency(final.loc[selected_mask, "Ocupacao"], occupation),
        "raw_rejected_or_surplus": _value_frequency(raw.loc[~selected_mask, "Ocupacao"], occupation),
        "final_rejected_or_surplus": _value_frequency(final.loc[~selected_mask, "Ocupacao"], occupation),
        "appeared_in_rejected_or_surplus_candidates": bool((raw.loc[~selected_mask, "Ocupacao"].astype(str) == occupation).any()),
        "removed_by_postprocessing": bool(
            (raw.loc[selected_mask, "Ocupacao"].astype(str) == occupation).any()
            and not (final.loc[selected_mask, "Ocupacao"].astype(str) == occupation).any()
        ),
    }


def run_ctgan_income_v3_raw_validity_diagnostic(
    confirmation_benchmark_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Executa diagnóstico da CTGAN candidate_c usando apenas artefatos treinados existentes."""
    benchmark_dir = Path(confirmation_benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_yaml_config(benchmark_dir / "benchmark_config.yaml")
    summary = pd.read_csv(benchmark_dir / "run_summary.csv")
    metadata = default_metadata()
    batch_size = int(config["generation"]["batch_size"])
    max_batches = int(config["generation"]["max_batches"])
    reference_date = str(config["benchmark"].get("reference_date", "2026-07-26"))
    target_rows = int(config["benchmark"]["synthetic_rows"])
    per_seed: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    cooccurrence_rows: list[dict[str, Any]] = []
    postprocessing_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    field_change_payload: dict[str, Any] = {"seeds": []}
    intersection_payload: dict[str, Any] = {"seeds": []}

    for _, run in summary.iterrows():
        seed = int(run["seed"])
        run_id = str(run["run_id"])
        run_dir = Path("artifacts") / "runs" / run_id / str(run["status"])
        if not run_dir.exists():
            run_dir = Path("artifacts") / "runs" / run_id / "approved"
        train = pd.read_parquet(run_dir / "train.parquet")
        holdout = pd.read_parquet(run_dir / "holdout.parquet")
        model_dir = Path("artifacts") / "models" / "ctgan" / run_id / "model"
        set_global_seed(seed, seed_tensorflow=False, seed_torch=True)
        synthesizer = CTGANSynthesizer.load(model_dir)
        candidates = generate_diagnostic_candidates(
            synthesizer=synthesizer,
            target_rows=target_rows,
            metadata=metadata,
            seed=seed,
            reference_date=reference_date,
            batch_size=batch_size,
            max_batches=max_batches,
            date_format=str(config["generation"]["date_format"]),
        )
        selected_raw = candidates.raw_candidates.loc[candidates.selected_indices].reset_index(drop=True)
        selected_final = candidates.final_candidates.loc[candidates.selected_indices].reset_index(drop=True)
        rule_report = diagnose_rule_validity(selected_raw, metadata)
        for item in rule_report["rows"]:
            rule_rows.append({"seed": seed, "run_id": run_id, **item})
        intersection = rule_report["intersections"]
        intersection_payload["seeds"].append({"seed": seed, "run_id": run_id, **intersection})
        for row in intersection["cooccurrence_rows"]:
            cooccurrence_rows.append({"seed": seed, "run_id": run_id, **row})
        changes = postprocessing_field_changes(
            candidates.raw_candidates,
            candidates.final_candidates,
            candidates.selected_indices,
            global_valid_mask=candidates.global_valid_mask,
        )
        field_change_payload["seeds"].append({"seed": seed, "run_id": run_id, **changes})
        for item in changes["field_summaries"]:
            postprocessing_rows.append({"seed": seed, "run_id": run_id, **item})
        for item in changes["transition_rows"]:
            transition_rows.append({"seed": seed, "run_id": run_id, **item})
        occupation_absence = _missing_occupations(train, holdout, selected_raw, selected_final)
        missing_details = [
            missing_occupation_diagnostic(
                occupation,
                train,
                holdout,
                candidates.raw_candidates,
                candidates.final_candidates,
                candidates.selected_indices,
            )
            for occupation in occupation_absence
        ]
        raw_validation = validate_profile_dataframe(selected_raw, metadata=metadata, final=False, reference_date=reference_date).report
        final_validation = validate_profile_dataframe(selected_final, metadata=metadata, final=True, reference_date=reference_date).report
        water = wasserstein_income_diagnostic(holdout, selected_raw, selected_final)
        dominant = _dominant_elementary_rule(rule_report["rows"])
        selected_count = int(len(selected_final))
        per_seed.append(
            {
                "seed": seed,
                "run_id": run_id,
                "model_artifact": str(model_dir),
                "raw_structural_validity_rate": _rate(raw_validation["valid_rows"], raw_validation["n_rows"]),
                "final_structural_validity_rate": _rate(final_validation["valid_rows"], final_validation["n_rows"]),
                "candidate_acceptance_rate": candidates.accounting.get("batch_acceptance_rate"),
                "global_acceptance_rate": candidates.accounting.get("global_acceptance_rate"),
                "postprocessing_rejection_rate": _rate(
                    int(candidates.accounting.get("rejected_by_global_rules", 0)),
                    int(candidates.accounting.get("total_candidates", 0)),
                ),
                "rule_most_reducing_raw_validity": {
                    "rule_id": dominant.get("rule_id"),
                    "rule_label": dominant.get("rule_label"),
                    "invalid": dominant.get("invalid"),
                    "failure_rate": dominant.get("failure_rate"),
                },
                "raw_geographic_validity_rate": _rule_rate(rule_report["rows"], "geographic_joint"),
                "raw_professional_validity_rate": _rule_rate(rule_report["rows"], "professional_joint"),
                "raw_non_relational_validity_rate": _rule_rate(rule_report["rows"], "non_relational_joint"),
                "postprocessing_classification": changes["classification"],
                "missing_occupations": missing_details,
                "income_distance": water,
                "raw_validation": raw_validation,
                "final_validation": final_validation,
                "generation_accounting": candidates.accounting,
            }
        )

    by_rule_path = output_dir / "ctgan_income_v3_raw_validity_by_rule.csv"
    postprocessing_path = output_dir / "ctgan_income_v3_postprocessing_summary.csv"
    cooccurrence_path = output_dir / "raw_rule_failure_cooccurrence.csv"
    transition_path = output_dir / "postprocessing_transition_summary.csv"
    pd.DataFrame(rule_rows).to_csv(by_rule_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(postprocessing_rows).to_csv(postprocessing_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(cooccurrence_rows).to_csv(cooccurrence_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(transition_rows).to_csv(transition_path, index=False, encoding="utf-8-sig")

    diagnostics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_confirmation_benchmark": str(benchmark_dir),
        "model": "ctgan",
        "profile": "ctgan_income_v3_recommended_candidate",
        "income_model_version": 3,
        "categorical_vocabulary_version": 2,
        "target_rows": target_rows,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "metric_semantics": metric_semantics(),
        "postprocessing_dependency_classification": classify_postprocessing_dependency(per_seed),
        "per_seed": per_seed,
        "files": {
            "metric_semantics": "metric_semantics.json",
            "by_rule": by_rule_path.name,
            "intersections": "raw_rule_failure_intersections.json",
            "cooccurrence": cooccurrence_path.name,
            "field_changes": "postprocessing_field_changes.json",
            "transitions": transition_path.name,
            "postprocessing_summary": postprocessing_path.name,
        },
    }
    write_json(metric_semantics(), output_dir / "metric_semantics.json")
    write_json(intersection_payload, output_dir / "raw_rule_failure_intersections.json")
    write_json(field_change_payload, output_dir / "postprocessing_field_changes.json")
    write_json(diagnostics, output_dir / "ctgan_income_v3_raw_validity_diagnostic.json")
    return diagnostics


def generate_diagnostic_candidates(
    synthesizer: Any,
    target_rows: int,
    metadata: DatasetMetadata,
    seed: int,
    reference_date: str,
    batch_size: int,
    max_batches: int,
    date_format: str,
) -> DiagnosticCandidates:
    """Reexecuta apenas a geração de candidatos para diagnóstico, sem treinamento."""
    fake = criar_faker(seed)
    rng = random.Random(seed)
    used = criar_estado_identificadores()
    raw_batches: list[pd.DataFrame] = []
    final_batches: list[pd.DataFrame] = []
    masks: list[pd.Series] = []
    attempts = 0
    sampling_seconds = 0.0
    postprocessing_seconds = 0.0
    validation_seconds = 0.0
    full_validation = None
    for attempts in range(1, int(max_batches) + 1):
        started = time.perf_counter()
        raw = synthesizer.sample(int(batch_size)).reset_index(drop=True)
        sampling_seconds += time.perf_counter() - started
        started = time.perf_counter()
        final = finalizar_perfis_sinteticos(
            raw,
            fake=fake,
            referencia=datetime.strptime(reference_date, "%Y-%m-%d"),
            rng=rng,
            date_format=date_format,
            used_identifiers=used,
        )
        postprocessing_seconds += time.perf_counter() - started
        started = time.perf_counter()
        validation = validate_profile_dataframe(final, metadata=metadata, final=True, reference_date=reference_date)
        validation_seconds += time.perf_counter() - started
        raw_batches.append(raw)
        final_batches.append(final)
        masks.append(validation.valid_mask.reset_index(drop=True))
        if int(pd.concat(masks, ignore_index=True).sum()) >= int(target_rows):
            all_final = pd.concat(final_batches, ignore_index=True)
            started = time.perf_counter()
            full_validation = validate_profile_dataframe(all_final, metadata=metadata, final=True, reference_date=reference_date)
            validation_seconds += time.perf_counter() - started
            if int(full_validation.valid_mask.sum()) >= int(target_rows):
                break
    all_raw = pd.concat(raw_batches, ignore_index=True) if raw_batches else pd.DataFrame()
    all_final = pd.concat(final_batches, ignore_index=True) if final_batches else pd.DataFrame()
    batch_mask = pd.concat(masks, ignore_index=True) if masks else pd.Series(dtype=bool)
    if full_validation is None:
        full_validation = validate_profile_dataframe(all_final, metadata=metadata, final=True, reference_date=reference_date)
    selection = select_valid_candidates(
        all_final,
        full_validation.valid_mask,
        n_target=int(target_rows),
        rejection_reasons=full_validation.report.get("reason_counts", {}),
        attempts=attempts,
        batch_valid_mask=batch_mask,
    )
    accounting = dict(selection.accounting)
    accounting["sampling_seconds"] = float(sampling_seconds)
    accounting["postprocessing_seconds"] = float(postprocessing_seconds)
    accounting["candidate_validation_seconds"] = float(validation_seconds)
    return DiagnosticCandidates(
        raw_candidates=all_raw,
        final_candidates=all_final,
        selected_indices=[int(index) for index in accounting.get("selected_candidate_indices", [])],
        accounting=accounting,
        global_valid_mask=full_validation.valid_mask.reset_index(drop=True),
        candidate_validation=full_validation.report,
    )


def classify_postprocessing_dependency(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Classifica dependência do pós-processamento por critérios explícitos."""
    if not per_seed:
        return {"classification": "indeterminada", "criteria": {}}
    raw_rates = [float(item["raw_structural_validity_rate"]) for item in per_seed]
    rejection_rates = [float(item["postprocessing_rejection_rate"]) for item in per_seed]
    repair_rates = [float(item["postprocessing_classification"]["repaired_rows"]) / max(float(item["postprocessing_classification"]["selected_rows"]), 1.0) for item in per_seed]
    max_raw = max(raw_rates)
    mean_rejection = sum(rejection_rates) / len(rejection_rates)
    mean_repair = sum(repair_rates) / len(repair_rates)
    if max_raw < 0.05 or mean_rejection >= 0.20:
        classification = "crítica" if mean_rejection >= 0.20 else "alta"
    elif max_raw < 0.50 or mean_repair >= 0.50:
        classification = "alta"
    elif max_raw < 0.80 or mean_repair >= 0.20:
        classification = "moderada"
    else:
        classification = "baixa"
    return {
        "classification": classification,
        "criteria": {
            "max_raw_structural_validity_rate": max_raw,
            "mean_postprocessing_rejection_rate": mean_rejection,
            "mean_repaired_selected_rate": mean_repair,
            "low": "raw >= 0.80 e repair < 0.20",
            "moderate": "raw >= 0.50 ou repair entre 0.20 e 0.50",
            "high": "raw < 0.50 ou repair >= 0.50",
            "critical": "rejection >= 0.20",
        },
    }


def _known_category_mask(
    frame: pd.DataFrame,
    metadata: DatasetMetadata,
    columns: tuple[str, ...] | None = None,
) -> pd.Series:
    target_columns = columns or tuple(
        column
        for column in DIAGNOSTIC_MODEL_COLUMNS
        if metadata.columns[column].categories and (metadata.columns[column].kind == "categorical" or metadata.columns[column].discrete)
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


def _numeric_domain_mask(frame: pd.DataFrame, column: str, minimum: float, maximum: float, integer: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = values.notna() & values.ge(minimum) & values.le(maximum)
    if integer:
        mask &= values.dropna().mod(1).reindex(frame.index, fill_value=1).eq(0)
    return mask.fillna(False).astype(bool)


def _region_state_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: region_for_state(str(row["Estado"])) == str(row["Regiao"]), axis=1).astype(bool)


def _state_municipality_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: str(row["Municipio"]) in STATE_MUNICIPALITIES.get(str(row["Estado"]), ()), axis=1).astype(bool)


def _state_ddd_mask(frame: pd.DataFrame) -> pd.Series:
    def valid(row: pd.Series) -> bool:
        try:
            return int(row["DDD"]) in STATE_DDDS.get(str(row["Estado"]), ())
        except (TypeError, ValueError):
            return False

    return frame.apply(valid, axis=1).astype(bool)


def _occupation_education_mask(frame: pd.DataFrame) -> pd.Series:
    def valid(row: pd.Series) -> bool:
        profile = get_occupation_profile(str(row["Ocupacao"]))
        return bool(profile is not None and str(row["Escolaridade"]) in profile.allowed_education)

    return frame.apply(valid, axis=1).astype(bool)


def _occupation_age_mask(frame: pd.DataFrame) -> pd.Series:
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


def _marital_age_mask(frame: pd.DataFrame) -> pd.Series:
    def valid(row: pd.Series) -> bool:
        try:
            age = int(row["Idade"])
        except (TypeError, ValueError):
            return False
        return not (str(row["Estado_Civil"]) == "Viúvo" and age < 25)

    return frame.apply(valid, axis=1).astype(bool)


def _rule_summary(frame: pd.DataFrame, rule_id: str, mask: pd.Series) -> dict[str, Any]:
    total = int(len(frame))
    valid = int(mask.sum())
    invalid_mask = ~mask.astype(bool)
    return {
        "rule_id": rule_id,
        "rule_label": RULE_LABELS[rule_id],
        "columns": list(RULE_COLUMNS[rule_id]),
        "total": total,
        "valid": valid,
        "invalid": int(invalid_mask.sum()),
        "validity_rate": _rate(valid, total),
        "failure_rate": _rate(int(invalid_mask.sum()), total),
        "top_invalid_combinations": _top_invalid_combinations(frame, rule_id, invalid_mask),
    }


def _top_invalid_combinations(frame: pd.DataFrame, rule_id: str, invalid_mask: pd.Series, limit: int = 10) -> list[dict[str, Any]]:
    subset = frame.loc[invalid_mask, list(RULE_COLUMNS[rule_id])].copy()
    if subset.empty:
        return []
    if rule_id in {"age_domain", "income_domain", "dependents_domain"}:
        column = RULE_COLUMNS[rule_id][0]
        labels = subset[column].apply(lambda value: _numeric_failure_label(value, column))
        counts = labels.value_counts().head(limit)
        return [{"combination": str(label), "count": int(count)} for label, count in counts.items()]
    if rule_id == "known_categories":
        rows = []
        metadata = default_metadata()
        for column in DIAGNOSTIC_MODEL_COLUMNS:
            if column not in subset.columns or not metadata.columns[column].categories:
                continue
            meta = metadata.columns[column]
            if meta.discrete:
                valid = pd.to_numeric(subset[column], errors="coerce").isin(meta.categories)
            else:
                valid = subset[column].astype(str).isin([str(category) for category in meta.categories])
            invalid_values = subset.loc[~valid.fillna(False), column].astype(str)
            for value, count in invalid_values.value_counts().head(limit).items():
                rows.append({"combination": f"{column}={value}", "count": int(count)})
        return sorted(rows, key=lambda item: item["count"], reverse=True)[:limit]
    labels = subset.astype(str).agg(" | ".join, axis=1)
    counts = labels.value_counts().head(limit)
    return [{"combination": str(label), "count": int(count)} for label, count in counts.items()]


def _numeric_failure_label(value: Any, column: str) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return f"{column}=não numérico"
    bounds = {
        "Idade": (18, 85),
        "Renda": (800.0, 50000.0),
        "Dependentes": (0, 6),
    }[column]
    if numeric < bounds[0]:
        return f"{column}=abaixo_do_mínimo"
    if numeric > bounds[1]:
        return f"{column}=acima_do_máximo"
    if column in {"Idade", "Dependentes"} and float(numeric) % 1 != 0:
        return f"{column}=não_inteiro"
    return f"{column}=outro"


def _canonical_compare_series(series: pd.Series, column: str) -> pd.Series:
    if column == "Renda":
        return pd.to_numeric(series, errors="coerce").round(2).astype("string").fillna("<NA>").astype(str)
    if column in {"Idade", "Dependentes", "DDD"}:
        numeric = pd.to_numeric(series, errors="coerce").round()
        return numeric.astype("Int64").astype("string").fillna("<NA>").astype(str)
    return series.astype("string").fillna("<NA>").astype(str)


def _missing_occupations(train: pd.DataFrame, holdout: pd.DataFrame, raw_selected: pd.DataFrame, final_selected: pd.DataFrame) -> list[str]:
    reference = set(train["Ocupacao"].astype(str)).union(set(holdout["Ocupacao"].astype(str)))
    final = set(final_selected["Ocupacao"].astype(str))
    raw = set(raw_selected["Ocupacao"].astype(str))
    return sorted(reference - final if reference - final else reference - raw)


def _value_frequency(series: pd.Series, value: str) -> dict[str, Any]:
    total = int(len(series))
    count = int((series.astype(str) == value).sum()) if total else 0
    return {"count": count, "total": total, "frequency": _rate(count, total)}


def _rule_rate(rows: list[dict[str, Any]], rule_id: str) -> float | None:
    for row in rows:
        if row["rule_id"] == rule_id:
            return row["validity_rate"]
    return None


def _dominant_elementary_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {"geographic_joint", "professional_joint", "non_relational_joint", "structural_global"}
    candidates = [row for row in rows if row["rule_id"] not in excluded]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: int(row["invalid"]))


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator <= 0:
        return None
    return float(float(numerator) / denominator)


def canonical_row_hash(values: dict[str, Any]) -> str:
    """Calcula hash determinístico de uma combinação sem expor a linha completa."""
    text = "|".join(f"{key}={values.get(key, '<NA>')}" for key in sorted(values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
