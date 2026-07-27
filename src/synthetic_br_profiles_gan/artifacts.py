"""Diretórios de artefatos e auxiliares de exportação."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.config import ConfigDict, save_yaml_config
from synthetic_br_profiles_gan.manifest import write_json


@dataclass(frozen=True)
class RunArtifactPaths:
    """Caminhos importantes de uma execução."""

    run_dir: Path
    approved_dir: Path
    quarantine_dir: Path
    status_dir: Path


def prepare_run_directories(artifacts_root: str | Path, run_id: str, status: str) -> RunArtifactPaths:
    """Cria os diretórios de execução, aprovação e quarentena."""
    run_dir = Path(artifacts_root) / "runs" / run_id
    approved_dir = run_dir / "approved"
    quarantine_dir = run_dir / "quarantine"
    approved_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    status_dir = approved_dir if status == "approved" else quarantine_dir
    status_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifactPaths(run_dir=run_dir, approved_dir=approved_dir, quarantine_dir=quarantine_dir, status_dir=status_dir)


def model_artifact_dir(artifacts_root: str | Path, model_name: str, run_id: str) -> Path:
    """Retorna o diretório de artefato de um modelo."""
    path = Path(artifacts_root) / "models" / model_name / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dataset(dataset: pd.DataFrame, output_dir: str | Path, export_xlsx: bool = True) -> dict[str, Path]:
    """Exporta um dataset em Parquet e, opcionalmente, XLSX."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {"dataset_parquet": output_path / "dataset.parquet"}
    dataset.to_parquet(paths["dataset_parquet"], index=False)
    if export_xlsx:
        paths["dataset_xlsx"] = output_path / "dataset.xlsx"
        dataset.to_excel(paths["dataset_xlsx"], index=False)
    return paths


def save_run_artifacts(
    dataset: pd.DataFrame,
    validation: dict[str, Any],
    evaluation: dict[str, Any],
    gate_result: dict[str, Any],
    manifest: dict[str, Any],
    config: ConfigDict,
    paths: RunArtifactPaths,
    export_xlsx: bool = True,
) -> dict[str, Path]:
    """Salva todos os artefatos da execução em approved ou quarantine."""
    artifact_paths = export_dataset(dataset, paths.status_dir, export_xlsx=export_xlsx)
    artifact_paths["validation"] = write_json(validation, paths.status_dir / "validation.json")
    artifact_paths["evaluation"] = write_json(evaluation, paths.status_dir / "evaluation.json")
    artifact_paths["quality_gates"] = write_json(gate_result, paths.status_dir / "quality_gates.json")
    artifact_paths["manifest"] = write_json(manifest, paths.status_dir / "manifest.json")
    artifact_paths["root_manifest"] = write_json(manifest, paths.run_dir / "manifest.json")
    artifact_paths["config"] = save_yaml_config(config, paths.status_dir / "config.yaml")
    artifact_paths["root_config"] = save_yaml_config(config, paths.run_dir / "config.yaml")
    return artifact_paths
