"""Command line interface for the synthetic profile pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import pandas as pd

from synthetic_br_profiles_gan.calibration import save_calibration_splits
from synthetic_br_profiles_gan.config import ConfigDict, deep_merge, load_yaml_config
from synthetic_br_profiles_gan.evaluation.metrics import evaluate_against_reference
from synthetic_br_profiles_gan.manifest import build_run_id, write_json
from synthetic_br_profiles_gan.metadata import DatasetMetadata, default_metadata
from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN
from synthetic_br_profiles_gan.pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    create_calibration,
    generate_profiles,
    run_pipeline,
    train_synthesizer,
)
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


def _load_model(model_name: str, model_path: str | Path):
    normalized = model_name.lower().replace("-", "_")
    path = Path(model_path)
    if normalized == "programmatic":
        return ProgrammaticSynthesizer.load(path)
    if normalized in {"simple_gan", "simple_tabular_gan", "dense_tabular_gan"}:
        return SimpleTabularGAN.load(path)
    if normalized in {"ctgan", "ctgan_synthesizer"}:
        return CTGANSynthesizer.load(path)
    raise ValueError(f"Unknown model: {model_name}")


def _load_config(path: str | None) -> ConfigDict:
    return load_yaml_config(path) if path else {}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Synthetic Brazilian profile generation pipeline.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("create-calibration", help="Create calibration/train/holdout data.")
    calibration.add_argument("--config", default="configs/calibration.yaml")
    calibration.add_argument("--output", default=None)

    train = subparsers.add_parser("train", help="Train a synthesizer.")
    train.add_argument("--model", required=True, choices=["programmatic", "simple_gan", "ctgan"])
    train.add_argument("--config", default=None)
    train.add_argument("--calibration", default=None, help="Path to train.parquet. Generated if omitted.")
    train.add_argument("--output", default=None)

    generate = subparsers.add_parser("generate", help="Generate final profiles from a saved model.")
    generate.add_argument("--model", required=True, choices=["programmatic", "simple_gan", "ctgan"])
    generate.add_argument("--model-path", default=None)
    generate.add_argument("--rows", type=int, required=True)
    generate.add_argument("--config", default=None)
    generate.add_argument("--output", default=None)

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
    return parser


def command_create_calibration(args: argparse.Namespace) -> int:
    """Handle create-calibration."""
    config = _load_config(args.config)
    output = Path(args.output) if args.output else Path("artifacts") / "calibration" / build_run_id()
    result = create_calibration(config.get("calibration", config), output)
    LOGGER.info("calibration_created", extra={"output": str(output), "paths": result["paths"]})
    return 0


def command_train(args: argparse.Namespace) -> int:
    """Handle train."""
    config = _load_config(args.config)
    metadata = default_metadata()
    if args.calibration:
        train = _read_table(args.calibration)
    else:
        calibration = create_calibration(config.get("calibration", config))
        train = calibration["train"]
    output = Path(args.output) if args.output else Path("artifacts") / "models" / args.model / build_run_id() / "model"
    model_config = config.get("models", {}).get(args.model, config)
    train_synthesizer(args.model, train, metadata, model_config, output)
    LOGGER.info("model_trained", extra={"model": args.model, "output": str(output)})
    return 0


def command_generate(args: argparse.Namespace) -> int:
    """Handle generate."""
    config = deep_merge(DEFAULT_PIPELINE_CONFIG, _load_config(args.config))
    metadata = default_metadata()
    if args.model_path:
        synthesizer = _load_model(args.model, args.model_path)
    else:
        synthesizer = ProgrammaticSynthesizer(config.get("models", {}).get("programmatic", {}))
        synthesizer.fit(pd.DataFrame(columns=metadata.model_columns), metadata)
    output = Path(args.output) if args.output else Path("artifacts") / "generated" / build_run_id()
    output.mkdir(parents=True, exist_ok=True)
    dataset, generation, candidate_validation = generate_profiles(
        synthesizer=synthesizer,
        n_target=args.rows,
        metadata=metadata,
        seed=int(config["seed"]),
        reference_date=str(config["reference_date"]),
        batch_size=int(config["generation"]["batch_size"]),
        max_batches=int(config["generation"]["max_batches"]),
        date_format=str(config["generation"]["date_format"]),
    )
    dataset_path = output / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    write_json({"generation": generation, "candidate_validation": candidate_validation}, output / "generation.json")
    LOGGER.info("dataset_generated", extra={"output": str(dataset_path), "rows": len(dataset)})
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Handle evaluate."""
    metadata = default_metadata()
    reference = _read_table(args.reference)
    synthetic = _read_table(args.synthetic)
    report = evaluate_against_reference(reference, synthetic, metadata)
    output = Path(args.output) if args.output else Path(args.synthetic).with_name("evaluation.json")
    write_json(report, output)
    LOGGER.info("evaluation_written", extra={"output": str(output)})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Handle validate."""
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
    """Handle full pipeline."""
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
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
