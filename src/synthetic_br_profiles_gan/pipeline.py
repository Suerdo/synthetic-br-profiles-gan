"""Serviços de pipeline para experimentos com perfis sintéticos brasileiros."""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.artifacts import (
    export_dataset,
    model_artifact_dir,
    prepare_run_directories,
)
from synthetic_br_profiles_gan.calibration import (
    DEFAULT_CALIBRATION_CONFIG,
    generate_calibration_dataset,
    save_calibration_splits,
    split_train_holdout,
)
from synthetic_br_profiles_gan.config import (
    ConfigDict,
    deep_merge,
    save_yaml_config,
    validate_calibration_config,
    validate_model_config,
    validate_pipeline_config,
)
from synthetic_br_profiles_gan.evaluation.metrics import evaluate_synthetic_data
from synthetic_br_profiles_gan.evaluation.quality_gates import DEFAULT_QUALITY_GATES, evaluate_quality_gates
from synthetic_br_profiles_gan.exceptions import QualityGateError
from synthetic_br_profiles_gan.generation import select_valid_candidates
from synthetic_br_profiles_gan.generators.demographics import (
    IDENTIFIER_COLUMNS,
    criar_estado_identificadores,
    criar_faker,
    finalizar_perfis_sinteticos,
)
from synthetic_br_profiles_gan.manifest import build_manifest, build_run_id, write_json
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.models.base import create_synthesizer
from synthetic_br_profiles_gan.models.preprocessing import DataPreprocessor
from synthetic_br_profiles_gan.reports.execution import exportar_resultados
from synthetic_br_profiles_gan.utils.reproducibility import seed_state_to_dict, set_global_seed
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe

LOGGER = logging.getLogger(__name__)


DEFAULT_PIPELINE_CONFIG: ConfigDict = {
    "seed": 41,
    "artifacts_root": "artifacts",
    "reference_date": "2026-07-26",
    "model": "programmatic",
    "calibration": DEFAULT_CALIBRATION_CONFIG,
    "models": {
        "programmatic": {},
        "simple_gan": {
            "seed": 41,
            "latent_dim": 16,
            "epochs": 10,
            "batch_size": 64,
            "verbose_every": 5,
            "metrics_every": 5,
        },
        "ctgan": {
            "seed": 41,
            "epochs": 10,
            "batch_size": 100,
            "verbose": False,
            "enable_gpu": False,
            "cuda": None,
        },
    },
    "generation": {
        "rows": 1000,
        "batch_size": 1024,
        "max_batches": 20,
        "date_format": "%Y-%m-%d",
    },
    "evaluation": {
        "privacy": {
            "max_nearest_neighbor_rows": 1000,
            "exclude_columns": ["Nome", "Data_Nascimento", "CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"],
        },
        "income_realism": {
            "minimum_group_rows": 30,
        },
    },
    "quality_gates": DEFAULT_QUALITY_GATES,
    "export": {"xlsx": True, "primary_format": "parquet"},
}


def create_calibration(config: ConfigDict | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Cria a calibração e os splits de treino e holdout."""
    effective = deep_merge(DEFAULT_CALIBRATION_CONFIG, config or {})
    validate_calibration_config(effective)
    metadata = default_metadata()
    df = generate_calibration_dataset(config=effective)
    train, holdout = split_train_holdout(
        df,
        holdout_fraction=float(effective["holdout_fraction"]),
        seed=int(effective["seed"]),
    )
    paths = None
    if output_dir is not None:
        paths = save_calibration_splits(df, train, holdout, output_dir, metadata=metadata)
    return {"calibration": df, "train": train, "holdout": holdout, "metadata": metadata, "paths": paths}


def _resolved_model_config(model_name: str, config: ConfigDict) -> ConfigDict:
    normalized = model_name.lower().replace("-", "_")
    if normalized == "programmatic":
        return deep_merge(DEFAULT_CALIBRATION_CONFIG, config)
    if normalized in {"simple_gan", "simple_tabular_gan", "dense_tabular_gan"}:
        from synthetic_br_profiles_gan.models.simple_gan import DEFAULT_SIMPLE_GAN_CONFIG

        return deep_merge(DEFAULT_SIMPLE_GAN_CONFIG, config)
    if normalized in {"ctgan", "ctgan_synthesizer"}:
        from synthetic_br_profiles_gan.models.ctgan import DEFAULT_CTGAN_CONFIG

        return deep_merge(DEFAULT_CTGAN_CONFIG, config)
    return config


def train_synthesizer(
    model_name: str,
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    config: ConfigDict | None = None,
    output_dir: str | Path | None = None,
):
    """Treina um sintetizador usando somente o split de treino."""
    model_config = _resolved_model_config(model_name, config or {})
    validate_model_config(model_name, model_config)
    synthesizer = create_synthesizer(model_name, model_config)
    LOGGER.info("training_synthesizer", extra={"model": model_name, "rows": len(train)})
    synthesizer.fit(train[metadata.model_columns], metadata)
    if output_dir is not None:
        synthesizer.save(Path(output_dir))
    return synthesizer


def generate_profiles(
    synthesizer,
    n_target: int,
    metadata: DatasetMetadata,
    seed: int,
    reference_date: str,
    batch_size: int = 1024,
    max_batches: int = 20,
    date_format: str = "%Y-%m-%d",
    return_raw: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]] | tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, pd.DataFrame]]:
    """Gera perfis finais e contabiliza linhas aceitas, rejeitadas e excedentes."""
    fake = criar_faker(seed)
    rng = random.Random(seed)
    candidates: list[pd.DataFrame] = []
    raw_candidates: list[pd.DataFrame] = []
    valid_masks: list[pd.Series] = []
    batch_lengths: list[int] = []
    attempts = 0
    sampling_seconds = 0.0
    postprocessing_seconds = 0.0
    candidate_validation_seconds = 0.0
    used_identifiers = criar_estado_identificadores()
    full_validation = None

    for attempts in range(1, int(max_batches) + 1):
        stage_started = time.perf_counter()
        core = synthesizer.sample(int(batch_size))
        sampling_seconds += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        final = finalizar_perfis_sinteticos(
            core,
            fake=fake,
            referencia=datetime.strptime(reference_date, "%Y-%m-%d"),
            rng=rng,
            date_format=date_format,
            used_identifiers=used_identifiers,
        )
        postprocessing_seconds += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        validation = validate_profile_dataframe(final, metadata=metadata, final=True, reference_date=reference_date)
        candidate_validation_seconds += time.perf_counter() - stage_started
        candidates.append(final)
        raw_candidates.append(core.reset_index(drop=True))
        batch_lengths.append(int(len(final)))
        valid_masks.append(validation.valid_mask.reset_index(drop=True))
        accepted_so_far = int(pd.concat(valid_masks, ignore_index=True).sum())
        if accepted_so_far >= n_target:
            stage_started = time.perf_counter()
            current_candidates = pd.concat(candidates, ignore_index=True)
            full_validation = validate_profile_dataframe(
                current_candidates,
                metadata=metadata,
                final=True,
                reference_date=reference_date,
            )
            candidate_validation_seconds += time.perf_counter() - stage_started
            if int(full_validation.valid_mask.sum()) >= n_target:
                break

    all_candidates = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    all_raw_candidates = pd.concat(raw_candidates, ignore_index=True) if raw_candidates else pd.DataFrame()
    all_masks = pd.concat(valid_masks, ignore_index=True) if valid_masks else pd.Series(dtype=bool)
    if full_validation is None or len(full_validation.valid_mask) != len(all_candidates):
        full_validation = validate_profile_dataframe(
            all_candidates,
            metadata=metadata,
            final=True,
            reference_date=reference_date,
        )
    selection = select_valid_candidates(
        all_candidates,
        full_validation.valid_mask,
        n_target=n_target,
        rejection_reasons=full_validation.report.get("reason_counts", {}),
        attempts=attempts,
        batch_valid_mask=all_masks,
    )
    if int(selection.accounting["selected"]) < int(n_target):
        raise RuntimeError(
            "Unable to generate the requested number of globally valid profiles "
            f"after {attempts} batches: selected={selection.accounting['selected']} target={int(n_target)}."
        )
    disagreeing_mask = all_masks.reindex(all_candidates.index).fillna(False).astype(bool) & ~full_validation.valid_mask.reindex(
        all_candidates.index
    ).fillna(False).astype(bool)
    selection.accounting["per_batch_valid_mask_count"] = int(all_masks.sum())
    selection.accounting["concatenated_valid_mask_count"] = int(all_masks.sum())
    selection.accounting["global_valid_mask_count"] = int(full_validation.valid_mask.sum())
    selection.accounting["global_mask_disagreeing_count"] = int(disagreeing_mask.sum())
    selection.accounting["global_mask_disagreeing_indices"] = [int(index) for index in all_candidates.index[disagreeing_mask][:100]]
    selection.accounting["cross_batch_identifier_duplicates"] = _count_cross_batch_identifier_duplicates(
        all_candidates,
        batch_lengths=batch_lengths,
    )
    selection.accounting["sampling_seconds"] = float(sampling_seconds)
    selection.accounting["postprocessing_seconds"] = float(postprocessing_seconds)
    selection.accounting["candidate_validation_seconds"] = float(candidate_validation_seconds)
    if return_raw:
        selected_indices = selection.accounting.get("selected_candidate_indices", [])
        raw_selected = all_raw_candidates.loc[selected_indices].reset_index(drop=True) if selected_indices else pd.DataFrame()
        return selection.selected, selection.accounting, full_validation.report, {
            "all_candidates": all_raw_candidates,
            "selected": raw_selected,
        }
    return selection.selected, selection.accounting, full_validation.report


def _count_cross_batch_identifier_duplicates(candidates: pd.DataFrame, batch_lengths: list[int]) -> int:
    if candidates.empty or not batch_lengths:
        return 0
    batch_for_index: dict[int, int] = {}
    start = 0
    for batch, length in enumerate(batch_lengths):
        for index in range(start, start + int(length)):
            batch_for_index[index] = batch
        start += int(length)
    duplicate_count = 0
    for column in IDENTIFIER_COLUMNS:
        if column not in candidates.columns:
            continue
        duplicated = candidates[column].duplicated(keep=False)
        if not duplicated.any():
            continue
        for _, group in candidates.loc[duplicated, [column]].groupby(column, sort=False):
            batches = {batch_for_index.get(int(index)) for index in group.index}
            if len(batches) > 1:
                duplicate_count += max(len(group) - 1, 0)
    return int(duplicate_count)


def _validation_rate(report: dict[str, Any]) -> float:
    rows = int(report.get("n_rows", 0) or 0)
    if rows <= 0:
        return 0.0
    return float(int(report.get("valid_rows", 0) or 0) / rows)


def _max_categorical_tvd(evaluation: dict[str, Any]) -> float | None:
    categorical = evaluation.get("against_holdout", {}).get("categorical", {})
    values = [
        float(metrics["total_variation_distance"])
        for metrics in categorical.values()
        if isinstance(metrics, dict) and metrics.get("total_variation_distance") is not None
    ]
    return max(values) if values else None


def _conditional_income_distance(evaluation: dict[str, Any]) -> float | None:
    summary = evaluation.get("conditional_income", {}).get("summary", {})
    value = summary.get("mean_conditional_income_wasserstein")
    return None if value is None else float(value)


def _raw_final_comparison(
    raw_validation: dict[str, Any],
    final_validation: dict[str, Any],
    raw_evaluation: dict[str, Any],
    final_evaluation: dict[str, Any],
    generation_accounting: dict[str, Any],
) -> dict[str, Any]:
    raw_validity = _validation_rate(raw_validation)
    final_validity = _validation_rate(final_validation)
    raw_tvd = _max_categorical_tvd(raw_evaluation)
    final_tvd = _max_categorical_tvd(final_evaluation)
    raw_income = _conditional_income_distance(raw_evaluation)
    final_income = _conditional_income_distance(final_evaluation)
    total_candidates = int(generation_accounting.get("total_candidates", 0) or 0)
    rejected = int(generation_accounting.get("rejected_by_global_rules", generation_accounting.get("rejected_by_rules", 0)) or 0)
    return {
        "raw_structural_validity_rate": raw_validity,
        "final_structural_validity_rate": final_validity,
        "postprocessing_repair_rate": float(max(final_validity - raw_validity, 0.0)),
        "postprocessing_rejection_rate": float(0.0 if total_candidates <= 0 else rejected / total_candidates),
        "categorical_tvd_raw": raw_tvd,
        "categorical_tvd_final": final_tvd,
        "conditional_income_distance_raw": raw_income,
        "conditional_income_distance_final": final_income,
        "raw_final_distribution_shift": (
            None if raw_tvd is None or final_tvd is None else float(abs(float(final_tvd) - float(raw_tvd)))
        ),
        "raw_reason_counts": raw_validation.get("reason_counts", {}),
        "final_reason_counts": final_validation.get("reason_counts", {}),
        "generation_accounting": {
            "total_candidates": total_candidates,
            "accepted_by_batch_rules": generation_accounting.get("accepted_by_batch_rules"),
            "accepted_by_global_rules": generation_accounting.get("accepted_by_global_rules"),
            "rejected_by_global_rules": generation_accounting.get("rejected_by_global_rules"),
            "selected": generation_accounting.get("selected"),
            "batch_acceptance_rate": generation_accounting.get("batch_acceptance_rate"),
            "global_acceptance_rate": generation_accounting.get("global_acceptance_rate"),
        },
        "interpretation": (
            "Indicadores diagnósticos para distinguir a saída bruta do sintetizador do resultado "
            "final pós-processado. Correções não são penalizadas automaticamente."
        ),
    }


def run_pipeline_on_splits(
    config: ConfigDict | None = None,
    model_name: str | None = None,
    train: pd.DataFrame | None = None,
    holdout: pd.DataFrame | None = None,
    metadata: DatasetMetadata | None = None,
    require_approved: bool = False,
    started_at_utc: datetime | None = None,
    resource_probe: Callable[[], float | None] | None = None,
) -> dict[str, Any]:
    """Executa treino, geração, validação, avaliação, gates e exportação nos splits fornecidos."""
    if train is None or holdout is None:
        raise ValueError("run_pipeline_on_splits requires train and holdout dataframes.")
    started = started_at_utc or datetime.now(timezone.utc)
    effective = deep_merge(DEFAULT_PIPELINE_CONFIG, config or {})
    selected_model = model_name or str(effective.get("model", "programmatic"))
    effective["model"] = selected_model
    validate_pipeline_config(effective)
    seed = int(effective["seed"])
    seed_state = set_global_seed(
        seed,
        seed_tensorflow=selected_model in {"simple_gan", "simple_tabular_gan", "dense_tabular_gan"},
        seed_torch=selected_model in {"ctgan", "ctgan_synthesizer"},
    )
    run_id = build_run_id(started)
    metadata = metadata or default_metadata()
    reference_date = str(effective["reference_date"])
    artifacts_root = Path(effective["artifacts_root"])
    requested_rows = int(effective["generation"]["rows"])
    stage_durations: dict[str, float] = {}
    stage_resources: dict[str, float | None] = {}

    LOGGER.info("pipeline_started", extra={"run_id": run_id, "model": selected_model, "seed": seed})

    default_model_seed = seed + 100_003 if selected_model == "programmatic" else seed
    model_config = deep_merge({"seed": default_model_seed}, effective.get("models", {}).get(selected_model, {}))
    if selected_model == "programmatic":
        model_config = deep_merge(effective["calibration"], model_config)
    model_config = _resolved_model_config(selected_model, model_config)
    effective.setdefault("models", {})[selected_model] = model_config
    model_dir = model_artifact_dir(artifacts_root, selected_model, run_id) / "model"
    if resource_probe is not None:
        stage_resources["memory_before_training_mb"] = resource_probe()
    stage_started = time.perf_counter()
    synthesizer = train_synthesizer(selected_model, train, metadata, config=model_config, output_dir=model_dir)
    stage_durations["training_seconds"] = float(time.perf_counter() - stage_started)
    if resource_probe is not None:
        stage_resources["memory_after_training_mb"] = resource_probe()

    generation_config = effective["generation"]
    if resource_probe is not None:
        stage_resources["memory_before_generation_mb"] = resource_probe()
    stage_started = time.perf_counter()
    dataset, generation_accounting, candidate_validation, raw_generation = generate_profiles(
        synthesizer=synthesizer,
        n_target=requested_rows,
        metadata=metadata,
        seed=seed,
        reference_date=reference_date,
        batch_size=int(generation_config["batch_size"]),
        max_batches=int(generation_config["max_batches"]),
        date_format=str(generation_config["date_format"]),
        return_raw=True,
    )
    stage_durations["generation_seconds"] = float(time.perf_counter() - stage_started)
    if resource_probe is not None:
        stage_resources["memory_after_generation_mb"] = resource_probe()
    stage_started = time.perf_counter()
    validation = validate_profile_dataframe(dataset, metadata=metadata, final=True, reference_date=reference_date).report
    raw_selected = raw_generation.get("selected", pd.DataFrame())
    raw_candidates = raw_generation.get("all_candidates", pd.DataFrame())
    raw_selected_validation = validate_profile_dataframe(raw_selected, metadata=metadata, final=False, reference_date=reference_date).report
    raw_candidate_validation = validate_profile_dataframe(raw_candidates, metadata=metadata, final=False, reference_date=reference_date).report
    stage_durations["validation_seconds"] = float(time.perf_counter() - stage_started)
    stage_started = time.perf_counter()
    raw_evaluation = evaluate_synthetic_data(
        raw_selected,
        train,
        holdout,
        metadata,
        max_nearest_neighbor_rows=int(effective.get("evaluation", {}).get("privacy", {}).get("max_nearest_neighbor_rows", 1000)),
        minimum_income_group_rows=int(effective.get("evaluation", {}).get("income_realism", {}).get("minimum_group_rows", 30)),
    )
    evaluation = evaluate_synthetic_data(
        dataset,
        train,
        holdout,
        metadata,
        max_nearest_neighbor_rows=int(effective.get("evaluation", {}).get("privacy", {}).get("max_nearest_neighbor_rows", 1000)),
        minimum_income_group_rows=int(effective.get("evaluation", {}).get("income_realism", {}).get("minimum_group_rows", 30)),
    )
    stage_durations["evaluation_seconds"] = float(time.perf_counter() - stage_started)
    raw_final_comparison = _raw_final_comparison(
        raw_validation=raw_selected_validation,
        final_validation=validation,
        raw_evaluation=raw_evaluation,
        final_evaluation=evaluation,
        generation_accounting=generation_accounting,
    )
    stage_started = time.perf_counter()
    gates = evaluate_quality_gates(validation, evaluation, effective["quality_gates"])
    stage_durations["quality_gates_seconds"] = float(time.perf_counter() - stage_started)

    status = gates.status
    paths = prepare_run_directories(artifacts_root, run_id, status)
    stage_started = time.perf_counter()
    artifact_paths = export_dataset(dataset, paths.status_dir, export_xlsx=bool(effective["export"].get("xlsx", True)))
    validation_path = write_json(validation, paths.status_dir / "validation.json")
    evaluation_path = write_json(evaluation, paths.status_dir / "evaluation.json")
    raw_evaluation_path = write_json(
        {
            "stage": "raw",
            "evaluation": raw_evaluation,
            "selected_validation": raw_selected_validation,
            "candidate_validation": raw_candidate_validation,
            "interpretation": (
                "Métricas calculadas nas colunas-base antes de normalização final, pós-processamento "
                "e criação dos campos derivados. O dataset raw completo não é persistido."
            ),
        },
        paths.status_dir / "raw_evaluation.json",
    )
    final_evaluation_path = write_json(
        {
            "stage": "final",
            "evaluation": evaluation,
            "validation": validation,
            "interpretation": "Métricas calculadas após normalização, pós-processamento e validação estrutural final.",
        },
        paths.status_dir / "final_evaluation.json",
    )
    raw_final_comparison_path = write_json(raw_final_comparison, paths.status_dir / "raw_final_comparison.json")
    privacy = evaluation.get("privacy", {})
    conditional_income = evaluation.get("conditional_income", {})
    memorization_path = write_json(
        {
            "columns_used": privacy.get("columns_used", []),
            "duplicate_base_rows": privacy.get("duplicate_base_rows"),
            "exact_matches": privacy.get("exact_matches"),
            "partial_matches": privacy.get("partial_matches"),
            "nearest_neighbor_train": privacy.get("nearest_neighbor_train"),
            "nearest_neighbor_holdout": privacy.get("nearest_neighbor_holdout"),
            "interpretation": (
                "Indicadores de diversidade e memorização calculados nas colunas-base. "
                "Identificadores derivados são excluídos da análise."
            ),
        },
        paths.status_dir / "memorization_metrics.json",
    )
    duplicate_base_rows_path = write_json(
        {
            "columns_used": privacy.get("columns_used", []),
            "summary": privacy.get("duplicate_base_rows", {}),
            "largest_groups": (privacy.get("duplicate_base_rows") or {}).get("largest_groups", []),
        },
        paths.status_dir / "duplicate_base_rows.json",
    )
    exact_train_matches_path = write_json(
        privacy.get("exact_matches", {}).get("train", {}),
        paths.status_dir / "exact_train_matches.json",
    )
    exact_holdout_matches_path = write_json(
        privacy.get("exact_matches", {}).get("holdout", {}),
        paths.status_dir / "exact_holdout_matches.json",
    )
    income_summary_rows = conditional_income.get("summary_rows", []) if isinstance(conditional_income, dict) else []
    income_comparison_rows = conditional_income.get("comparison_rows", []) if isinstance(conditional_income, dict) else []
    income_tail_rows = conditional_income.get("tail_events", []) if isinstance(conditional_income, dict) else []
    income_summary_csv = paths.status_dir / "conditional_income_summary.csv"
    income_summary_parquet = paths.status_dir / "conditional_income_summary.parquet"
    income_comparison_csv = paths.status_dir / "conditional_income_comparison.csv"
    income_tail_csv = paths.status_dir / "conditional_income_tail_events.csv"
    pd.DataFrame(income_summary_rows).to_csv(income_summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(income_summary_rows).to_parquet(income_summary_parquet, index=False)
    pd.DataFrame(income_comparison_rows).to_csv(income_comparison_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(income_tail_rows).to_csv(income_tail_csv, index=False, encoding="utf-8-sig")
    income_plausibility_path = write_json(
        {
            "summary": conditional_income.get("summary", {}) if isinstance(conditional_income, dict) else {},
            "minimum_group_rows": conditional_income.get("minimum_group_rows") if isinstance(conditional_income, dict) else None,
            "groupings": conditional_income.get("groupings", {}) if isinstance(conditional_income, dict) else {},
            "interpretation": conditional_income.get("interpretation") if isinstance(conditional_income, dict) else None,
        },
        paths.status_dir / "income_plausibility_summary.json",
    )
    gates_payload = {
        "status": gates.status,
        "failures": gates.failures,
        "metrics_checked": gates.metrics_checked,
    }
    gates_path = write_json(gates_payload, paths.status_dir / "quality_gates.json")
    config_path = save_yaml_config(effective, paths.status_dir / "config.yaml")
    root_config_path = save_yaml_config(effective, paths.run_dir / "config.yaml")
    generation_path = write_json(
        {
            "generation_accounting": generation_accounting,
            "candidate_validation": candidate_validation,
            "raw_candidate_validation": raw_candidate_validation,
            "raw_selected_validation": raw_selected_validation,
            "raw_final_comparison": raw_final_comparison,
        },
        paths.status_dir / "generation.json",
    )
    metadata_path = metadata.save(paths.status_dir / "metadata.json")
    train_path = paths.status_dir / "train.parquet"
    holdout_path = paths.status_dir / "holdout.parquet"
    train.to_parquet(train_path, index=False)
    holdout.to_parquet(holdout_path, index=False)
    stage_durations["export_seconds"] = float(time.perf_counter() - stage_started)

    ended = datetime.now(timezone.utc)
    manifest_paths = {
        **artifact_paths,
        "validation": validation_path,
        "evaluation": evaluation_path,
        "raw_evaluation": raw_evaluation_path,
        "final_evaluation": final_evaluation_path,
        "raw_final_comparison": raw_final_comparison_path,
        "memorization_metrics": memorization_path,
        "duplicate_base_rows": duplicate_base_rows_path,
        "exact_train_matches": exact_train_matches_path,
        "exact_holdout_matches": exact_holdout_matches_path,
        "conditional_income_summary_csv": income_summary_csv,
        "conditional_income_summary_parquet": income_summary_parquet,
        "conditional_income_comparison_csv": income_comparison_csv,
        "conditional_income_tail_events_csv": income_tail_csv,
        "income_plausibility_summary": income_plausibility_path,
        "quality_gates": gates_path,
        "generation": generation_path,
        "config": config_path,
        "metadata": metadata_path,
        "train": train_path,
        "holdout": holdout_path,
    }
    manifest = build_manifest(
        run_id=run_id,
        model=selected_model,
        seed=seed,
        requested_rows=requested_rows,
        generated_rows=len(dataset),
        status=status,
        config=effective,
        artifact_paths=manifest_paths,
        started_at_utc=started,
        ended_at_utc=ended,
        root=Path.cwd(),
    )
    manifest["seed_state"] = seed_state_to_dict(seed_state)
    manifest["income_model_version"] = int(effective.get("calibration", {}).get("income_model_version", 1))
    manifest["generation_accounting"] = generation_accounting
    manifest["quality_gate_failures"] = gates.failures
    manifest["stage_durations_seconds"] = stage_durations
    if stage_resources:
        manifest["stage_resources"] = stage_resources
    manifest_path = write_json(manifest, paths.status_dir / "manifest.json")
    root_manifest_path = write_json(manifest, paths.run_dir / "manifest.json")
    manifest_paths["manifest"] = manifest_path
    manifest_paths["root_manifest"] = root_manifest_path
    manifest_paths["root_config"] = root_config_path

    result = {
        "run_id": run_id,
        "status": status,
        "dataset": dataset,
        "validation": validation,
        "evaluation": evaluation,
        "quality_gates": gates_payload,
        "generation": generation_accounting,
        "manifest": manifest,
        "stage_durations": stage_durations,
        "stage_resources": stage_resources,
        "paths": manifest_paths,
        "model_dir": model_dir,
    }
    LOGGER.info("pipeline_finished", extra={"run_id": run_id, "status": status})
    if require_approved and status != "approved":
        raise QualityGateError(f"Run {run_id} finished with status={status}.")
    return result


def run_pipeline(
    config: ConfigDict | None = None,
    model_name: str | None = None,
    require_approved: bool = False,
) -> dict[str, Any]:
    """Executa calibração, treino, geração, validação, avaliação, gates e exportação."""
    started = datetime.now(timezone.utc)
    effective = deep_merge(DEFAULT_PIPELINE_CONFIG, config or {})
    selected_model = model_name or str(effective.get("model", "programmatic"))
    effective["model"] = selected_model
    validate_pipeline_config(effective)
    calibration = create_calibration(effective["calibration"])
    return run_pipeline_on_splits(
        config=effective,
        model_name=selected_model,
        train=calibration["train"],
        holdout=calibration["holdout"],
        metadata=calibration["metadata"],
        require_approved=require_approved,
        started_at_utc=started,
    )


def gerar_sinteticos_com_metricas(
    generator,
    discriminator,
    preprocessor: DataPreprocessor,
    latent_dim: int,
    n_target: int = 1000,
    batch_gen: int = 2048,
    score_threshold: float = 0.50,
    max_batches: int = 200,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Geração legada de candidatos sem aceitação por limiar do discriminador.

    ``score_threshold`` é mantido apenas para diagnósticos retrocompatíveis.
    A aceitação se baseia em regras estruturais e de domínio, não em realismo calibrado.
    """
    started = time.perf_counter()
    accepted_scaled: list[np.ndarray] = []
    accepted_orig: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    valid_masks: list[pd.Series] = []
    discriminator_scores: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()

    for batch_index in range(int(max_batches)):
        noise = np.random.normal(0, 1, (batch_gen, latent_dim))
        generated_scaled = generator.predict(noise, verbose=0)
        scores = discriminator.predict(generated_scaled, verbose=0).reshape(-1)
        discriminator_scores.append(scores)
        generated = preprocessor.inverse_transform(generated_scaled)

        if {"Idade", "Sexo", "Renda"}.issubset(generated.columns):
            mask = (
                generated["Idade"].between(18, 65)
                & generated["Renda"].between(1200, 25000)
                & generated["Sexo"].between(0, 1)
            )
            rejection_counts["dominio_invalido"] += int((~mask).sum())
        else:
            validation = validate_profile_dataframe(generated, metadata=preprocessor.metadata, final=False)
            mask = validation.valid_mask
            rejection_counts.update(validation.report.get("reason_counts", {}))

        candidate_frames.append(generated)
        valid_masks.append(mask.reset_index(drop=True))
        if int(pd.concat(valid_masks, ignore_index=True).sum()) >= n_target:
            accepted_scaled.append(generated_scaled[mask.to_numpy()])
            accepted_orig.append(generated.loc[mask])
            break
        accepted_scaled.append(generated_scaled[mask.to_numpy()])
        accepted_orig.append(generated.loc[mask])

    total_candidates = int(sum(len(frame) for frame in candidate_frames))
    if not accepted_orig or sum(len(frame) for frame in accepted_orig) == 0:
        raise RuntimeError("No candidate passed structural/domain validation.")

    all_accepted_orig = pd.concat(accepted_orig, ignore_index=True)
    all_accepted_scaled = np.vstack(accepted_scaled)
    selected_orig = all_accepted_orig.iloc[:n_target].copy()
    selected_scaled = all_accepted_scaled[:n_target]
    accepted_by_rules = int(len(all_accepted_orig))
    score_values = np.concatenate(discriminator_scores) if discriminator_scores else np.array([])
    duration = time.perf_counter() - started
    report = {
        "n_target": int(n_target),
        "score_threshold_diagnostic_only": float(score_threshold),
        "batch_gen": int(batch_gen),
        "max_batches": int(max_batches),
        "total_candidates": total_candidates,
        "accepted_by_rules": accepted_by_rules,
        "selected": int(len(selected_orig)),
        "accepted_but_not_selected": int(max(accepted_by_rules - len(selected_orig), 0)),
        "rejected_by_rules": int(total_candidates - accepted_by_rules),
        "attempts_or_batches": int(len(candidate_frames)),
        "real_acceptance_rate": float(0.0 if total_candidates == 0 else accepted_by_rules / total_candidates),
        "rejection_reasons": dict(rejection_counts),
        "tempo_geracao_seg": float(duration),
        "throughput_selecionados_por_seg": float(0.0 if duration == 0 else len(selected_orig) / duration),
        "discriminator_score_diagnostics": {
            "mean": float(score_values.mean()) if score_values.size else None,
            "min": float(score_values.min()) if score_values.size else None,
            "max": float(score_values.max()) if score_values.size else None,
        },
    }
    return selected_orig, selected_scaled, report


def executar_pipeline(
    n_target: int = 1000,
    seed: int = 41,
    output_dir: str | Path = "data/outputs",
    calibration_size: int = 20000,
    latent_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 64,
    batch_gen: int = 2048,
    score_threshold: float = 0.50,
    max_batches: int = 200,
    reference_date: datetime | None = None,
    model_name: str = "simple_gan",
) -> dict:
    """Ponto de entrada retrocompatível usado pelo script legado."""
    reference = (reference_date or datetime.now()).strftime("%Y-%m-%d")
    config = deep_merge(
        DEFAULT_PIPELINE_CONFIG,
        {
            "seed": seed,
            "artifacts_root": str(Path(output_dir) / "artifacts"),
            "reference_date": reference,
            "model": model_name,
            "calibration": {"seed": seed, "num_rows": calibration_size},
            "models": {
                "simple_gan": {
                    "seed": seed,
                    "latent_dim": latent_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                },
                "programmatic": {"seed": seed},
            },
            "generation": {
                "rows": n_target,
                "batch_size": batch_gen,
                "max_batches": max_batches,
                "date_format": "%Y-%m-%d",
            },
        },
    )
    result = run_pipeline(config=config, model_name=model_name)
    legacy_paths = exportar_resultados(
        result["dataset"],
        {
            "seed": seed,
            "n_target": n_target,
            "run_id": result["run_id"],
            "status": result["status"],
            "generation": result["generation"],
            "validation": result["validation"],
            "quality_gates": result["quality_gates"],
        },
        output_dir,
    )
    result["paths"]["legacy_dataset"] = legacy_paths["dataset"]
    result["paths"]["legacy_relatorio"] = legacy_paths["relatorio"]
    return result
