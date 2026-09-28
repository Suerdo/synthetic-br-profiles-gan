"""Endpoints de geração assíncrona."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from synthetic_br_profiles_gan.api.dependencies import get_job_manager, require_ui_session_id
from synthetic_br_profiles_gan.api.jobs import GenerationJob, GenerationJobManager
from synthetic_br_profiles_gan.api.schemas.generations import (
    GenerationCreateRequest,
    GenerationCreateResponse,
    GenerationManifestResponse,
    GenerationPreviewResponse,
    GenerationStatusResponse,
)

router = APIRouter(tags=["generations"])


@router.post("/generations", response_model=GenerationCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def create_generation(
    request: GenerationCreateRequest,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> GenerationCreateResponse:
    """Agenda uma geração sem aceitar caminhos locais do cliente."""
    job = manager.start_generation(
        session_id=session_id,
        model=request.model,
        artifact_id=request.artifact_id,
        num_rows=request.num_rows,
        output_format=request.output_format,
        seed=request.seed,
        selected_columns=request.selected_columns,
        column_preset=request.column_preset,
    )
    return GenerationCreateResponse(
        generation_id=job.generation_id,
        status=job.status,
        status_url=f"/api/generations/{job.generation_id}",
    )


@router.get("/generations/{generation_id}", response_model=GenerationStatusResponse)
def generation_status(
    generation_id: str,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> GenerationStatusResponse:
    """Consulta o estado de uma geração da sessão atual."""
    job = _get_job_or_404(manager, generation_id, session_id)
    return _status_response(job)


@router.get("/generations/{generation_id}/preview", response_model=GenerationPreviewResponse)
def generation_preview(
    generation_id: str,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> GenerationPreviewResponse:
    """Retorna uma amostra limitada do dataset gerado."""
    job = _get_job_or_404(manager, generation_id, session_id)
    preview = manager.read_preview(job)
    return GenerationPreviewResponse(**preview)


@router.get("/generations/{generation_id}/manifest", response_model=GenerationManifestResponse)
def generation_manifest(
    generation_id: str,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> GenerationManifestResponse:
    """Retorna o manifesto sanitizado em JSON."""
    job = _get_job_or_404(manager, generation_id, session_id)
    return GenerationManifestResponse(generation_id=generation_id, manifest=manager.read_manifest(job))


@router.get("/generations/{generation_id}/download/dataset")
def download_dataset(
    generation_id: str,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> FileResponse:
    """Baixa o dataset exportado pelo job."""
    job = _get_job_or_404(manager, generation_id, session_id)
    path = manager.dataset_path(job)
    return FileResponse(path, filename=_download_name(job, path))


@router.get("/generations/{generation_id}/download/manifest")
def download_manifest(
    generation_id: str,
    session_id: str = Depends(require_ui_session_id),
    manager: GenerationJobManager = Depends(get_job_manager),
) -> FileResponse:
    """Baixa o manifesto sanitizado da geração."""
    job = _get_job_or_404(manager, generation_id, session_id)
    path = manager.manifest_path(job)
    return FileResponse(path, filename=f"perfis-sinteticos-{job.model}-{job.num_rows}.manifest.json")


def _get_job_or_404(manager: GenerationJobManager, generation_id: str, session_id: str) -> GenerationJob:
    job = manager.get_job(generation_id, session_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Geração não encontrada para esta sessão.")
    return job


def _status_response(job: GenerationJob) -> GenerationStatusResponse:
    error = None
    if job.error is not None:
        error = {"type": job.error.get("type", "Error"), "message": job.error.get("message", "Falha na geração.")}
    completed_urls = job.status == "completed"
    return GenerationStatusResponse(
        generation_id=job.generation_id,
        status=job.status,
        model=job.model,
        artifact_id=job.artifact_id,
        num_rows=job.num_rows,
        output_format=job.output_format,
        seed=job.seed,
        submitted_at_utc=job.submitted_at_utc,
        started_at_utc=job.started_at_utc,
        completed_at_utc=job.completed_at_utc,
        duration_seconds=job.duration_seconds,
        exported_columns=list(job.exported_columns),
        internal_columns=list(job.internal_columns),
        validation=job.validation,
        dataset_download_url=f"/api/generations/{job.generation_id}/download/dataset" if completed_urls else None,
        manifest_download_url=f"/api/generations/{job.generation_id}/download/manifest" if completed_urls else None,
        preview_url=f"/api/generations/{job.generation_id}/preview" if completed_urls else None,
        error=error,
    )


def _download_name(job: GenerationJob, path: Path) -> str:
    return f"perfis-sinteticos-{job.model}-{job.num_rows}{path.suffix}"
