"""Interface de linha de comando para o pipeline de perfis sintéticos."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import pandas as pd

from synthetic_br_profiles_gan.benchmark import run_benchmark
from synthetic_br_profiles_gan.calibration import save_calibration_splits
from synthetic_br_profiles_gan.column_catalog import available_presets
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge, load_yaml_config
from synthetic_br_profiles_gan.evaluation.metrics import evaluate_against_reference
from synthetic_br_profiles_gan.manifest import build_run_id, write_json
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    create_calibration,
    run_pipeline,
)
from synthetic_br_profiles_gan.services.generation_service import GenerationRequest, run_generation
from synthetic_br_profiles_gan.services.training_service import TrainingRequest, run_training
from synthetic_br_profiles_gan.exceptions import ConfigurationError, ModelBackendUnavailable, PipelineError
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe

LOGGER = logging.getLogger(__name__)


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)
    raise ValueError(f"Unsupported table format: {table_path}")


def _load_config(path: str | None) -> ConfigDict:
    return load_yaml_config(path) if path else {}


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser da CLI."""
    parser = argparse.ArgumentParser(description="Synthetic Brazilian profile generation pipeline.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("create-calibration", help="Create calibration/train/holdout data.")
    calibration.add_argument("--config", default="configs/calibration.yaml")
    calibration.add_argument("--output", default=None)

    train = subparsers.add_parser("train", help="Train and save a reusable synthesizer artifact.")
    train.add_argument("--model", required=True, choices=["programmatic", "simple_gan", "ctgan"])
    train.add_argument("--config", default=None)
    train.add_argument("--calibration", default=None, help="Path to train.parquet. Generated if omitted.")
    train.add_argument("--output", default=None)
    train.add_argument("--train-rows", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--overwrite", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate final profiles from a saved model or direct programmatic generator.")
    generate.add_argument("--model", default=None, choices=["programmatic", "simple_gan", "ctgan"])
    generate.add_argument("--model-path", default=None)
    generate.add_argument("--rows", type=int, required=True)
    generate.add_argument("--config", default=None)
    generate.add_argument("--output", default=None)
    generate.add_argument("--format", choices=["csv", "json", "parquet"], default="parquet")
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--columns", nargs="+", default=None, help="Colunas finais a exportar, em ordem.")
    generate.add_argument(
        "--preset",
        default=None,
        help=f"Preset de colunas ({', '.join(available_presets())}).",
    )
    generate.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate synthetic data against one reference table.")
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--synthetic", required=True)
    evaluate.add_argument("--output", default=None)

    validate = subparsers.add_parser("validate", help="Validate a final synthetic dataset.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--config", default=None)
    validate.add_argument("--output", default=None)

    pipeline = subparsers.add_parser("pipeline", help="Run the full experimental pipeline.")
    pipeline.add_argument("--model", default=None, choices=["programmatic", "simple_gan", "ctgan"])
    pipeline.add_argument("--config", default="configs/pipeline.yaml")
    pipeline.add_argument("--require-approved", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run a reproducible synthesizer benchmark.")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")
    benchmark.add_argument("--models", nargs="+", choices=["programmatic", "simple_gan", "ctgan"], default=None)
    benchmark.add_argument("--seeds", nargs="+", type=int, default=None)
    benchmark.add_argument("--train-sizes", nargs="+", type=int, default=None)
    return parser


def command_create_calibration(args: argparse.Namespace) -> int:
    """Executa o comando create-calibration."""
    config = _load_config(args.config)
    output = Path(args.output) if args.output else Path("artifacts") / "calibration" / build_run_id()
    result = create_calibration(config.get("calibration", config), output)
    LOGGER.info("calibration_created", extra={"output": str(output), "paths": result["paths"]})
    return 0


def command_train(args: argparse.Namespace) -> int:
    """Executa o comando train."""
    config = _load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", config.get("calibration", {}).get("seed", 41)))
    holdout_fraction = float(config.get("holdout_fraction", config.get("calibration", {}).get("holdout_fraction", 0.20)))
    if args.train_rows is not None:
        train_rows = int(args.train_rows)
    elif "train_rows" in config:
        train_rows = int(config["train_rows"])
    elif "num_rows" in config.get("calibration", {}):
        train_rows = int(round(int(config["calibration"]["num_rows"]) * (1.0 - holdout_fraction)))
    else:
        train_rows = 1000
    output = Path(args.output) if args.output else Path("artifacts") / "models" / f"{args.model}-{build_run_id()}"
    result = run_training(
        TrainingRequest(
            model=args.model,
            output_path=output,
            config=config,
            seed=seed,
            train_rows=train_rows,
            holdout_fraction=holdout_fraction,
            overwrite=bool(args.overwrite),
            calibration_path=Path(args.calibration) if args.calibration else None,
        )
    )
    LOGGER.info(
        "model_trained",
        extra={
            "model": result.model,
            "output": str(result.output_path),
            "train_rows": result.train_rows,
            "holdout_rows": result.holdout_rows,
            "manifest": str(result.manifest_path),
        },
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    """Executa o comando generate."""
    config = _load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 41))
    if args.output:
        output = Path(args.output)
    else:
        output = Path("artifacts") / "generated" / f"{build_run_id()}.{args.format}"
    result = run_generation(
        GenerationRequest(
            model=args.model,
            model_path=Path(args.model_path) if args.model_path else None,
            num_rows=int(args.rows),
            output_path=output,
            output_format=str(args.format),
            seed=seed,
            config=config,
            overwrite=bool(args.overwrite),
            selected_columns=_parse_column_arguments(args.columns),
            column_preset=args.preset,
        )
    )
    LOGGER.info(
        "dataset_generated",
        extra={
            "model": result.model,
            "output": str(result.output_path),
            "manifest": str(result.manifest_path),
            "rows": result.num_rows,
        },
    )
    return 0


def _parse_column_arguments(values: list[str] | None) -> list[str] | None:
    """Converte argumentos de colunas separados por espaço ou vírgula."""
    if values is None:
        return None
    columns: list[str] = []
    for value in values:
        parts = str(value).split(",")
        columns.extend(part.strip() for part in parts if part.strip())
    return columns


def command_evaluate(args: argparse.Namespace) -> int:
    """Executa o comando evaluate."""
    metadata = default_metadata()
    reference = _read_table(args.reference)
    synthetic = _read_table(args.synthetic)
    report = evaluate_against_reference(reference, synthetic, metadata)
    output = Path(args.output) if args.output else Path(args.synthetic).with_name("evaluation.json")
    write_json(report, output)
    LOGGER.info("evaluation_written", extra={"output": str(output)})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Executa o comando validate."""
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, _load_config(args.config))
    dataset = _read_table(args.input)
    report = validate_profile_dataframe(
        dataset,
        metadata=default_metadata(),
        final=True,
        reference_date=str(config["reference_date"]),
    ).report
    output = Path(args.output) if args.output else Path(args.input).with_name("validation.json")
    write_json(report, output)
    LOGGER.info("validation_written", extra={"output": str(output), "invalid_rows": report["invalid_rows"]})
    return 0 if report["is_valid"] else 2


def command_pipeline(args: argparse.Namespace) -> int:
    """Executa o pipeline completo."""
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, _load_config(args.config))
    result = run_pipeline(config=config, model_name=args.model, require_approved=args.require_approved)
    LOGGER.info(
        "pipeline_complete",
        extra={
            "run_id": result["run_id"],
            "status": result["status"],
            "manifest": str(result["paths"]["manifest"]),
        },
    )
    return 0 if result["status"] == "approved" else 2 if args.require_approved else 0


def command_benchmark(args: argparse.Namespace) -> int:
    """Executa o comando benchmark."""
    config = _load_config(args.config)
    config.setdefault("benchmark", {})
    if args.models is not None:
        config["benchmark"]["models"] = args.models
    if args.seeds is not None:
        config["benchmark"]["seeds"] = args.seeds
    if args.train_sizes is not None:
        config["benchmark"]["train_sizes"] = args.train_sizes
        config["benchmark"].pop("calibration_rows", None)
    result = run_benchmark(config)
    LOGGER.info(
        "benchmark_complete",
        extra={
            "benchmark_id": result["benchmark_id"],
            "status": result["status"],
            "benchmark_dir": str(result["benchmark_dir"]),
            "completed_runs": result["completed_runs"],
            "failed_runs": result["failed_runs"],
        },
    )
    return 0 if result["status"] == "completed" else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s %(name)s %(message)s")
    commands = {
        "create-calibration": command_create_calibration,
        "train": command_train,
        "generate": command_generate,
        "evaluate": command_evaluate,
        "validate": command_validate,
        "pipeline": command_pipeline,
        "benchmark": command_benchmark,
    }
    try:
        return int(commands[args.command](args))
    except ModelBackendUnavailable as exc:
        LOGGER.error("%s", exc)
        return 2
    except ConfigurationError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 2
    except (OSError, json.JSONDecodeError, pickle.UnpicklingError, ValueError, RuntimeError) as exc:
        LOGGER.error("Command failed: %s", exc)
        return 2
