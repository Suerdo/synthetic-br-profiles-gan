"""Carregamento seguro de sintetizadores salvos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.exceptions import ModelSerializationError
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN


REQUIRED_MODEL_FILES = {
    "programmatic": ["config.json", "metadata.json"],
    "simple_gan": [
        "generator.keras",
        "discriminator.keras",
        "preprocessor.pkl",
        "metadata.json",
        "config.json",
        "training_history.json",
    ],
    "ctgan": ["model.pkl", "metadata.json", "metadata_ctgan.json"],
}


@dataclass(frozen=True)
class LoadedSynthesizer:
    """Sintetizador carregado e manifesto de treinamento associado."""

    model: str
    synthesizer: Any
    artifact_path: Path
    manifest: dict[str, Any]


def load_training_manifest(model_path: str | Path) -> dict[str, Any]:
    """Carrega e valida o manifesto de treinamento de um artefato de modelo."""
    artifact_path = Path(model_path)
    manifest_path = artifact_path / "training_manifest.json"
    if not manifest_path.exists():
        raise ModelSerializationError(f"Missing training_manifest.json in model artifact: {artifact_path}")
    try:
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSerializationError(f"Could not read training manifest from {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelSerializationError(f"Training manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise ModelSerializationError(f"Unsupported training manifest schema_version: {manifest.get('schema_version')}")
    if manifest.get("artifact_type") != "trained_synthesizer":
        raise ModelSerializationError(f"Invalid artifact_type in training manifest: {manifest.get('artifact_type')}")
    model = str(manifest.get("model") or "")
    if model not in REQUIRED_MODEL_FILES:
        raise ModelSerializationError(f"Unknown model in training manifest: {model}")
    metadata = default_metadata()
    if list(manifest.get("model_columns", [])) != metadata.model_columns:
        raise ModelSerializationError("Model artifact model_columns are incompatible with the current schema.")
    if list(manifest.get("final_columns", [])) != metadata.final_columns:
        raise ModelSerializationError("Model artifact final_columns are incompatible with the current schema.")
    return manifest


def validate_required_model_files(model_path: str | Path, model: str) -> None:
    """Confirma a presença dos arquivos obrigatórios para o tipo de modelo."""
    artifact_path = Path(model_path)
    missing = [name for name in REQUIRED_MODEL_FILES[model] if not (artifact_path / name).exists()]
    if missing:
        raise ModelSerializationError(f"Missing required file(s) for {model}: {', '.join(missing)}")


def load_saved_synthesizer(model_path: str | Path, expected_model: str | None = None) -> LoadedSynthesizer:
    """Carrega um sintetizador salvo usando o manifesto como fonte de verdade."""
    artifact_path = Path(model_path)
    if not artifact_path.exists():
        raise ModelSerializationError(f"Model artifact path does not exist: {artifact_path}")
    if not artifact_path.is_dir():
        raise ModelSerializationError(f"Model artifact path must be a directory: {artifact_path}")
    manifest = load_training_manifest(artifact_path)
    model = str(manifest["model"])
    if expected_model is not None and expected_model != model:
        raise ModelSerializationError(f"Requested model '{expected_model}' is incompatible with saved artifact model '{model}'.")
    validate_required_model_files(artifact_path, model)
    if model == "programmatic":
        synthesizer = ProgrammaticSynthesizer.load(artifact_path)
    elif model == "simple_gan":
        synthesizer = SimpleTabularGAN.load(artifact_path)
    elif model == "ctgan":
        synthesizer = CTGANSynthesizer.load(artifact_path)
    else:
        raise ModelSerializationError(f"Unsupported saved model: {model}")
    return LoadedSynthesizer(model=model, synthesizer=synthesizer, artifact_path=artifact_path, manifest=manifest)
