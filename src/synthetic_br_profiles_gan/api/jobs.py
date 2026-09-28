"""Gerenciamento assíncrono de gerações solicitadas pela interface web."""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import pandas as pd

from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.column_catalog import resolve_column_selection
from synthetic_br_profiles_gan.exceptions import ConfigurationError, PipelineError
from synthetic_br_profiles_gan.manifest import write_json
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS
from synthetic_br_profiles_gan.models.registry import (
    SavedModelArtifact,
    get_recommended_artifact,
    list_saved_model_artifacts,
    sort_artifacts_for_generation,
)
from synthetic_br_profiles_gan.services.generation_service import GenerationRequest, GenerationResult, run_generation


JOB_STATUSES = {"queued", "running", "completed", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class GenerationJob:
    """Estado em memória de uma geração iniciada pela interface React."""

    generation_id: str
    session_id: str
    model: str
    artifact_id: str | None
    num_rows: int
    output_format: str
    seed: int
    selected_columns: tuple[str, ...] | None
    column_preset: str | None
    output_path: Path
    manifest_path: Path
    preview_path: Path
    status: str = "queued"
    submitted_at_utc: str = field(default_factory=_utc_now)
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_seconds: float | None = None
    exported_columns: tuple[str, ...] = ()
    internal_columns: tuple[str, ...] = tuple(FINAL_COLUMNS)
    validation: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class GenerationJobManager:
    """Executor em memória para jobs curtos de geração da interface web."""

    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=int(settings.max_workers), thread_name_prefix="web-generation")
        self._jobs: dict[str, GenerationJob] = {}
        self._lock = Lock()

    def start_generation(
        self,
        *,
        session_id: str,
        model: str,
        artifact_id: str | None,
        num_rows: int,
        output_format: str,
        seed: int,
        selected_columns: list[str] | None,
        column_preset: str | None,
    ) -> GenerationJob:
        """Valida a solicitação e agenda a geração em background."""
        model = model.lower().replace("-", "_")
        self._validate_generation_payload(
            model=model,
            artifact_id=artifact_id,
            num_rows=num_rows,
            output_format=output_format,
            selected_columns=selected_columns,
            column_preset=column_preset,
        )
        resolved_selection = resolve_column_selection(
            tuple(selected_columns) if selected_columns is not None else None,
            preset=column_preset,
            available_columns=FINAL_COLUMNS,
        )
        generation_id = str(uuid4())
        extension = output_format.lower()
        output_dir = self._session_generation_dir(session_id, generation_id)
        output_path = output_dir / f"perfis-sinteticos-{model}-{int(num_rows)}.{extension}"
        manifest_path = output_path.with_suffix(".manifest.json")
        preview_path = output_dir / "preview.json"
        job = GenerationJob(
            generation_id=generation_id,
            session_id=session_id,
            model=model,
            artifact_id=artifact_id,
            num_rows=int(num_rows),
            output_format=output_format.lower(),
            seed=int(seed),
            selected_columns=tuple(selected_columns) if selected_columns is not None else None,
            column_preset=column_preset,
            output_path=output_path,
            manifest_path=manifest_path,
            preview_path=preview_path,
            exported_columns=tuple(resolved_selection.exported_columns),
        )
        with self._lock:
            self._jobs[generation_id] = job
        self._executor.submit(self._run_job, job)
        return job

    def get_job(self, generation_id: str, session_id: str) -> GenerationJob | None:
        """Retorna o job somente quando ele pertence à sessão informada."""
        with self._lock:
            job = self._jobs.get(generation_id)
        if job is None or job.session_id != session_id:
            return None
        return job

    def read_preview(self, job: GenerationJob) -> dict[str, Any]:
        """Lê a amostra segura criada ao final da geração."""
        if job.status != "completed":
            raise ConfigurationError("A prévia só está disponível após a conclusão da geração.")
        if not job.preview_path.exists():
            raise ConfigurationError("A prévia da geração não foi encontrada.")
        with job.preview_path.open(encoding="utf-8") as file:
            return json.load(file)

    def read_manifest(self, job: GenerationJob) -> dict[str, Any]:
        """Lê o manifesto sanitizado da geração."""
        if job.status != "completed":
            raise ConfigurationError("O manifesto só está disponível após a conclusão da geração.")
        if not job.manifest_path.exists():
            raise ConfigurationError("O manifesto da geração não foi encontrado.")
        with job.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        return _public_manifest(manifest, job.artifact_id)

    def dataset_path(self, job: GenerationJob) -> Path:
        """Retorna o caminho do dataset gerado após validar o estado do job."""
        if job.status != "completed" or not job.output_path.exists():
            raise ConfigurationError("O dataset gerado ainda não está disponível.")
        return job.output_path

    def manifest_path(self, job: GenerationJob) -> Path:
        """Retorna o caminho do manifesto após validar o estado do job."""
        if job.status != "completed" or not job.manifest_path.exists():
            raise ConfigurationError("O manifesto gerado ainda não está disponível.")
        return job.manifest_path

    def list_artifacts(self, model: str) -> list[SavedModelArtifact]:
        """Lista artefatos válidos ordenados para seleção pela interface."""
        return sort_artifacts_for_generation(list_saved_model_artifacts(self.settings.models_root, model=model))

    def recommended_artifact(self, model: str) -> SavedModelArtifact | None:
        """Retorna o artefato recomendado do modelo, quando existir."""
        return get_recommended_artifact(self.settings.models_root, model)

    def _run_job(self, job: GenerationJob) -> None:
        start = time.perf_counter()
        self._update_job(job.generation_id, status="running", started_at_utc=_utc_now())
        try:
            artifact = self._artifact_for_job(job)
            request = GenerationRequest(
                model=job.model if artifact is None else None,
                model_path=None if artifact is None else artifact.artifact_path,
                num_rows=job.num_rows,
                output_path=job.output_path,
                output_format=job.output_format,
                seed=job.seed,
                overwrite=False,
                selected_columns=job.selected_columns,
                column_preset=job.column_preset,
            )
            result = run_generation(request)
            self._write_preview(job, result)
            self._sanitize_manifest_file(job)
            self._update_job(
                job.generation_id,
                status="completed",
                completed_at_utc=_utc_now(),
                duration_seconds=float(time.perf_counter() - start),
                exported_columns=result.exported_columns,
                internal_columns=result.internal_columns,
                validation=result.validation_report,
            )
        except Exception as exc:  # pragma: no cover - o teste cobre o estado público, não o traceback.
            self._update_job(
                job.generation_id,
                status="failed",
                completed_at_utc=_utc_now(),
                duration_seconds=float(time.perf_counter() - start),
                error=_job_error(exc),
            )

    def _artifact_for_job(self, job: GenerationJob) -> SavedModelArtifact | None:
        if job.model == "programmatic":
            return None
        for artifact in self.list_artifacts(job.model):
            if artifact.artifact_id == job.artifact_id:
                return artifact
        raise ConfigurationError("Artefato de modelo não encontrado ou indisponível para esta aplicação.")

    def _write_preview(self, job: GenerationJob, result: GenerationResult) -> None:
        frame = _read_dataset_preview(result.output_path, job.output_format, self.settings.preview_rows)
        payload = {
            "generation_id": job.generation_id,
            "rows": _dataframe_records(frame),
            "columns": list(frame.columns),
            "preview_rows": int(self.settings.preview_rows),
            "total_rows": int(result.num_rows),
        }
        write_json(payload, job.preview_path)

    def _sanitize_manifest_file(self, job: GenerationJob) -> None:
        with job.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        write_json(_public_manifest(manifest, job.artifact_id), job.manifest_path)

    def _session_generation_dir(self, session_id: str, generation_id: str) -> Path:
        return self.settings.web_sessions_root / session_id / generation_id

    def _validate_generation_payload(
        self,
        *,
        model: str,
        artifact_id: str | None,
        num_rows: int,
        output_format: str,
        selected_columns: list[str] | None,
        column_preset: str | None,
    ) -> None:
        if model not in {"programmatic", "ctgan", "simple_gan"}:
            raise ConfigurationError(f"Modelo desconhecido: {model}")
        if int(num_rows) < self.settings.min_rows:
            raise ConfigurationError(f"A quantidade de registros deve ser no mínimo {self.settings.min_rows}.")
        if int(num_rows) > int(self.settings.row_limits[model]):
            raise ConfigurationError(
                f"A interface permite até {self.settings.row_limits[model]} registros para {model} nesta fase."
            )
        if output_format not in {"csv", "json", "parquet"}:
            raise ConfigurationError(f"Formato de saída não suportado: {output_format}")
        if model == "programmatic" and artifact_id:
            raise ConfigurationError("A geração programática direta não utiliza artifact_id.")
        if model in {"ctgan", "simple_gan"} and not artifact_id:
            raise ConfigurationError("Informe artifact_id para gerar com CTGAN ou GAN simples.")
        resolve_column_selection(
            tuple(selected_columns) if selected_columns is not None else None,
            preset=column_preset,
            available_columns=FINAL_COLUMNS,
        )

    def _update_job(self, generation_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[generation_id]
            for key, value in changes.items():
                setattr(job, key, value)


def _read_dataset_preview(path: Path, output_format: str, limit: int) -> pd.DataFrame:
    if output_format == "csv":
        return pd.read_csv(path, sep=";", nrows=limit)
    if output_format == "json":
        with path.open(encoding="utf-8") as file:
            rows = json.load(file)
        return pd.DataFrame(rows).head(limit)
    if output_format == "parquet":
        return pd.read_parquet(path).head(limit)
    raise ConfigurationError(f"Formato de saída não suportado: {output_format}")


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    payload = json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))
    return payload if isinstance(payload, list) else []


def _public_manifest(manifest: dict[str, Any], artifact_id: str | None) -> dict[str, Any]:
    public_manifest = dict(manifest)
    public_manifest["artifact_id"] = artifact_id
    for key in ["model_artifact", "output_file", "source_training_manifest", "environment", "git_commit"]:
        public_manifest.pop(key, None)
    return public_manifest


def _job_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, PipelineError):
        message = str(exc)
    else:
        message = "Falha inesperada durante a geração. Consulte os logs do servidor."
    return {
        "type": type(exc).__name__,
        "message": message,
        "traceback": traceback.format_exc(limit=8),
    }
