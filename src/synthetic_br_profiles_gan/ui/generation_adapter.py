"""Adaptador entre a interface e o serviço de geração."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from synthetic_br_profiles_gan.config import ConfigurationError
from synthetic_br_profiles_gan.models.registry import SavedModelArtifact, list_saved_model_artifacts
from synthetic_br_profiles_gan.services.generation_service import GenerationRequest, GenerationResult, run_generation
from synthetic_br_profiles_gan.ui.ui_config import UIConfig, SUPPORTED_UI_FORMATS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UIGenerationRequest:
    """Solicitação de geração originada pela interface."""

    model: str
    rows: int
    output_format: str
    seed: int
    config: UIConfig
    artifact_id: str | None = None
    selected_columns: Sequence[str] | None = None
    column_preset: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class UIGenerationResult:
    """Resultado da geração pronto para exibição e download."""

    service_result: GenerationResult
    dataset: pd.DataFrame
    manifest: dict[str, Any]
    session_dir: Path
    output_path: Path
    manifest_path: Path
    download_filename: str
    manifest_download_filename: str


def list_available_artifacts(models_root: str | Path) -> dict[str, list[SavedModelArtifact]]:
    """Lista artefatos válidos agrupados por modelo."""
    grouped = {"programmatic": [], "ctgan": [], "simple_gan": []}
    for artifact in list_saved_model_artifacts(models_root):
        grouped.setdefault(artifact.model, []).append(artifact)
    return grouped


def run_ui_generation(request: UIGenerationRequest) -> UIGenerationResult:
    """Valida a solicitação da interface e delega a geração ao serviço."""
    _validate_ui_request(request)
    artifact = _resolve_artifact(request)
    session_dir = _create_generation_dir(request.config.sessions_root, request.session_id)
    output_path = session_dir / _dataset_filename(request.model, request.rows, request.output_format)
    LOGGER.info(
        "ui_generation_started",
        extra={
            "model": request.model,
            "artifact_id": request.artifact_id,
            "rows": int(request.rows),
            "format": request.output_format,
        },
    )
    try:
        service_result = run_generation(
            GenerationRequest(
                model=request.model if artifact is None else None,
                model_path=None if artifact is None else artifact.artifact_path,
                num_rows=int(request.rows),
                output_path=output_path,
                output_format=request.output_format,
                seed=int(request.seed),
                selected_columns=None if request.selected_columns is None else list(request.selected_columns),
                column_preset=request.column_preset,
                overwrite=False,
            )
        )
        dataset = read_generated_dataset(service_result.output_path, request.output_format)
        manifest = json.loads(service_result.manifest_path.read_text(encoding="utf-8"))
        LOGGER.info(
            "ui_generation_finished",
            extra={
                "model": service_result.model,
                "rows": int(request.rows),
                "format": request.output_format,
                "duration_seconds": service_result.duration_seconds,
            },
        )
        return UIGenerationResult(
            service_result=service_result,
            dataset=dataset,
            manifest=manifest,
            session_dir=session_dir,
            output_path=service_result.output_path,
            manifest_path=service_result.manifest_path,
            download_filename=_dataset_filename(request.model, request.rows, request.output_format),
            manifest_download_filename=_manifest_filename(request.model, request.rows),
        )
    except Exception as exc:
        LOGGER.exception(
            "ui_generation_failed",
            extra={"model": request.model, "rows": int(request.rows), "format": request.output_format, "error_type": type(exc).__name__},
        )
        raise


def read_generated_dataset(path: str | Path, output_format: str) -> pd.DataFrame:
    """Lê um dataset exportado pelo serviço de geração."""
    dataset_path = Path(path)
    if output_format == "csv":
        return pd.read_csv(dataset_path, sep=";")
    if output_format == "json":
        return pd.DataFrame(json.loads(dataset_path.read_text(encoding="utf-8")))
    if output_format == "parquet":
        return pd.read_parquet(dataset_path)
    raise ConfigurationError(f"Formato de saída não suportado pela interface: {output_format}.")


def artifact_label(artifact: SavedModelArtifact) -> str:
    """Formata um rótulo curto para um artefato disponível."""
    created = artifact.created_at_utc or "data não registrada"
    train_rows = "treino não registrado" if artifact.train_rows is None else f"treino {artifact.train_rows}"
    seed = "seed não registrada" if artifact.seed is None else f"seed {artifact.seed}"
    vocabulary = f"vocabulário {artifact.categorical_vocabulary_version}"
    return f"{artifact.artifact_id} · {created} · {train_rows} · {seed} · {vocabulary}"


def validation_summary(validation_report: dict[str, Any]) -> list[tuple[str, bool]]:
    """Converte o relatório estrutural em itens amigáveis para a interface."""
    reason_counts = validation_report.get("reason_counts", {}) or {}
    return [
        ("Quantidade de linhas correta", bool(validation_report.get("is_valid", False))),
        ("Schema interno completo", not bool(validation_report.get("details", {}).get("missing_columns"))),
        ("CPFs estruturalmente válidos", int(reason_counts.get("CPF_invalido", 0)) == 0),
        ("CPFs sem duplicidade no lote", int(reason_counts.get("CPF_duplicado", 0)) == 0),
        ("Datas de nascimento coerentes com as idades", int(reason_counts.get("idade_data_nascimento_incompativel", 0)) == 0),
        ("Municípios coerentes com os estados", int(reason_counts.get("municipio_estado_incompativel", 0)) == 0),
        ("DDDs coerentes com os estados", int(reason_counts.get("ddd_estado_incompativel", 0)) == 0),
        ("Telefones no formato esperado", int(reason_counts.get("Telefone_invalido", 0)) == 0),
        ("Telefones coerentes com os DDDs", int(reason_counts.get("telefone_ddd_incompativel", 0)) == 0),
    ]


def _validate_ui_request(request: UIGenerationRequest) -> None:
    if request.model not in {"programmatic", "ctgan", "simple_gan"}:
        raise ConfigurationError("Modelo desconhecido para a interface.")
    if request.output_format not in SUPPORTED_UI_FORMATS:
        raise ConfigurationError("Formato de saída inválido.")
    rows = int(request.rows)
    if rows < int(request.config.min_rows):
        raise ConfigurationError(f"A quantidade deve ser pelo menos {request.config.min_rows}.")
    limit = int(request.config.limits[request.model])
    if rows > limit:
        raise ConfigurationError(
            f"A quantidade excede o limite operacional desta interface para {request.model}: {limit}."
        )
    if int(request.seed) < 0:
        raise ConfigurationError("A seed deve ser maior ou igual a zero.")
    if request.model == "programmatic" and request.artifact_id is not None:
        raise ConfigurationError("O modelo programático não exige artefato salvo.")
    if request.model in {"ctgan", "simple_gan"} and not request.artifact_id:
        raise ConfigurationError("Selecione um artefato de modelo válido antes de gerar.")


def _resolve_artifact(request: UIGenerationRequest) -> SavedModelArtifact | None:
    if request.model == "programmatic":
        return None
    artifacts = list_saved_model_artifacts(request.config.models_root, model=request.model)
    for artifact in artifacts:
        if artifact.artifact_id == request.artifact_id:
            return artifact
    raise ConfigurationError("Artefato de modelo indisponível ou fora do diretório administrado.")


def _create_generation_dir(root: Path, session_id: str | None) -> Path:
    safe_session = session_id or uuid.uuid4().hex
    generation_id = uuid.uuid4().hex
    output_dir = root / safe_session / generation_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _dataset_filename(model: str, rows: int, output_format: str) -> str:
    return f"perfis-sinteticos-{model}-{int(rows)}.{output_format}"


def _manifest_filename(model: str, rows: int) -> str:
    return f"perfis-sinteticos-{model}-{int(rows)}.manifest.json"
