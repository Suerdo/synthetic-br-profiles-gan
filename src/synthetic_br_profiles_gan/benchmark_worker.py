"""Subprocess worker for isolated operational capacity benchmark runs."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from synthetic_br_profiles_gan.benchmark import (
    _ctgan_training_inference,
    _mean_epoch_seconds,
    _path_size_mb,
    _read_simple_gan_history,
    _run_artifact_size_mb,
)
from synthetic_br_profiles_gan.config import load_yaml_config
from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable
from synthetic_br_profiles_gan.manifest import environment_info, write_json
from synthetic_br_profiles_gan.metadata import DatasetMetadata
from synthetic_br_profiles_gan.pipeline import run_pipeline_on_splits


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser."""
    parser = argparse.ArgumentParser(description="Run one isolated capacity benchmark model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-backend", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one model/tamanho combination and write a structured result."""
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    try:
        config = load_yaml_config(args.config)
        backend_warmup_seconds = _warmup_backend(args.model) if args.warmup_backend else 0.0
        metadata = DatasetMetadata.load(args.metadata)
        train = pd.read_parquet(args.train)
        holdout = pd.read_parquet(args.holdout)
        result = run_pipeline_on_splits(config=config, model_name=args.model, train=train, holdout=holdout, metadata=metadata)
        payload = _payload_from_pipeline_result(args.model, result, config, len(train))
        payload["backend_warmup_seconds"] = backend_warmup_seconds
        write_json(payload, output_path)
        return 0
    except ModelBackendUnavailable as exc:
        write_json(_error_payload(args.model, "backend_unavailable", "inicialização", exc), output_path)
        return 3
    except Exception as exc:
        write_json(_error_payload(args.model, "failed", "pipeline", exc), output_path)
        return 4


def _payload_from_pipeline_result(model: str, result: dict, config: dict, train_size: int) -> dict:
    quality_status = result.get("status")
    technical_status = _technical_status_for_quality_status(str(quality_status))
    history = _read_simple_gan_history(result.get("model_dir"))
    ctgan_training = _ctgan_training_inference(model, {**result, "resolved_run_config": config}, train_size)
    return {
        "technical_status": technical_status,
        "quality_status": quality_status,
        "run_id": result.get("run_id"),
        "duration_seconds": result.get("manifest", {}).get("duration_seconds"),
        "stage_durations": result.get("stage_durations", {}),
        "model_size_mb": _path_size_mb(result.get("model_dir")),
        "artifact_size_mb": _run_artifact_size_mb(result),
        "batches_per_epoch": history.get("batches_per_epoch"),
        "generator_updates": history.get("total_generator_updates"),
        "discriminator_updates": history.get("total_discriminator_updates"),
        "mean_epoch_seconds": _mean_epoch_seconds(history),
        "ctgan_batches_per_epoch_inferred": ctgan_training.get("batches_per_epoch_inferred"),
        "ctgan_total_batches_inferred": ctgan_training.get("total_batches_inferred"),
        "backend": model,
        "quality_gates": result.get("quality_gates", {}),
        "paths": {key: str(value) for key, value in result.get("paths", {}).items()},
        "environment": environment_info(),
    }


def _technical_status_for_quality_status(status: str) -> str:
    if status == "approved":
        return "completed"
    if status == "quarantined":
        return "quality_quarantined"
    if status == "rejected":
        return "quality_rejected"
    return "failed"


def _warmup_backend(model: str) -> float:
    started = time.perf_counter()
    if model == "simple_gan":
        from synthetic_br_profiles_gan.models.gan import _require_tensorflow

        _require_tensorflow()
    elif model == "ctgan":
        from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

        CTGANSynthesizer._ctgan_class()
    return float(time.perf_counter() - started)


def _error_payload(model: str, status: str, stage: str, exc: BaseException) -> dict:
    return {
        "technical_status": status,
        "quality_status": None,
        "run_id": None,
        "failure_stage": stage,
        "failure_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "backend": model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
    }


if __name__ == "__main__":
    sys.exit(main())
