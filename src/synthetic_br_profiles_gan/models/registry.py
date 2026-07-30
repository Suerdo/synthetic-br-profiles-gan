"""Carregamento seguro de sintetizadores salvos.

Artefatos com `pickle` ou formatos equivalentes devem vir de diretórios
produzidos ou previamente aprovados pela própria aplicação.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.domain.geography import LEGACY_GEOGRAPHY_MODEL_VERSION
from synthetic_br_profiles_gan.exceptions import ModelSerializationError
from synthetic_br_profiles_gan.localization import (
    CATEGORICAL_VOCABULARY_VERSION,
    DATA_LOCALE,
    INCOME_MODEL_VERSION,
    UNICODE_NORMALIZATION,
)
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


@dataclass(frozen=True)
class SavedModelArtifact:
    """Resumo de um artefato de modelo salvo e validado por manifesto."""

    model: str
    artifact_id: str
    artifact_path: Path
    created_at_utc: str | None
    train_rows: int | None
    seed: int | None
    schema_version: int
    training_required: bool
    model_size_bytes: int | None
    data_locale: str | None
    unicode_normalization: str | None
    categorical_vocabulary_version: int
    income_model_version: int
    is_legacy_income_model: bool
    purpose: str
    approval_status: str
    is_legacy_vocabulary: bool
    compatibility_normalization_required: bool
    manifest: dict[str, Any]
    geography_model_version: int = LEGACY_GEOGRAPHY_MODEL_VERSION
    geography_catalog_version: int | None = None
    geography_catalog_checksum: str | None = None
    is_legacy_geography_model: bool = False


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


def list_saved_model_artifacts(models_root: str | Path, model: str | None = None) -> list[SavedModelArtifact]:
    """Lista artefatos válidos dentro do diretório administrado de modelos."""
    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return []
    root_resolved = root.resolve()
    requested_model = model.lower().replace("-", "_") if model else None
    artifacts: list[SavedModelArtifact] = []
    for manifest_path in sorted(root.rglob("training_manifest.json")):
        artifact_path = manifest_path.parent
        try:
            resolved_artifact = artifact_path.resolve()
            if not resolved_artifact.is_relative_to(root_resolved):
                continue
            manifest = load_training_manifest(artifact_path)
            artifact_model = str(manifest["model"])
            if requested_model is not None and artifact_model != requested_model:
                continue
            validate_required_model_files(artifact_path, artifact_model)
        except ModelSerializationError:
            continue
        vocabulary_version = _optional_int(manifest.get("categorical_vocabulary_version")) or 1
        income_model_version = _optional_int(manifest.get("income_model_version")) or 1
        geography_model_version = _optional_int(manifest.get("geography_model_version")) or LEGACY_GEOGRAPHY_MODEL_VERSION
        artifacts.append(
            SavedModelArtifact(
                model=artifact_model,
                artifact_id=_artifact_id(root_resolved, resolved_artifact),
                artifact_path=resolved_artifact,
                created_at_utc=manifest.get("created_at_utc"),
                train_rows=_optional_int(manifest.get("train_rows")),
                seed=_optional_int(manifest.get("seed")),
                schema_version=int(manifest.get("schema_version", 0)),
                training_required=bool(manifest.get("training_required", artifact_model != "programmatic")),
                model_size_bytes=_optional_int(manifest.get("model_size_bytes")),
                data_locale=manifest.get("data_locale") or DATA_LOCALE,
                unicode_normalization=manifest.get("unicode_normalization") or UNICODE_NORMALIZATION,
                categorical_vocabulary_version=vocabulary_version,
                income_model_version=income_model_version,
                is_legacy_income_model=bool(income_model_version < INCOME_MODEL_VERSION),
                purpose=_artifact_purpose(manifest, artifact_path),
                approval_status=_artifact_approval_status(manifest, artifact_path),
                is_legacy_vocabulary=bool(vocabulary_version < CATEGORICAL_VOCABULARY_VERSION),
                compatibility_normalization_required=bool(vocabulary_version < CATEGORICAL_VOCABULARY_VERSION),
                manifest=manifest,
                geography_model_version=geography_model_version,
                geography_catalog_version=_optional_int(manifest.get("geography_catalog_version")),
                geography_catalog_checksum=manifest.get("geography_catalog_checksum"),
                is_legacy_geography_model=bool(geography_model_version < 2),
            )
        )
    artifacts.sort(key=lambda item: (item.model, item.created_at_utc or "", item.artifact_id))
    return artifacts


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


def _artifact_id(root: Path, artifact_path: Path) -> str:
    try:
        return str(artifact_path.relative_to(root))
    except ValueError:
        return artifact_path.name


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _artifact_purpose(manifest: dict[str, Any], artifact_path: Path) -> str:
    purpose = manifest.get("purpose") or manifest.get("artifact_purpose") or manifest.get("assessment_mode")
    if isinstance(purpose, str) and purpose:
        return purpose.lower().replace("-", "_")
    name = artifact_path.name.lower()
    if "smoke" in name:
        return "smoke"
    if "candidate" in name or "candidato" in name:
        return "candidate"
    if "experimental" in name or "experiment" in name:
        return "experimental"
    return "legacy" if _optional_int(manifest.get("categorical_vocabulary_version")) is None else "experimental"


def _artifact_approval_status(manifest: dict[str, Any], artifact_path: Path) -> str:
    status = manifest.get("approval_status") or manifest.get("model_status") or manifest.get("status")
    if isinstance(status, str) and status:
        return status.lower().replace("-", "_")
    purpose = _artifact_purpose(manifest, artifact_path)
    if purpose == "approved":
        return "approved"
    if purpose == "smoke":
        return "smoke"
    if purpose == "candidate":
        return "candidate"
    if purpose == "legacy":
        return "legacy"
    return "experimental"
