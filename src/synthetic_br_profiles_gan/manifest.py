"""Run IDs, hashes de arquivos e manifestos de execução."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from synthetic_br_profiles_gan.config import ConfigDict, config_hash


def build_run_id(timestamp: datetime | None = None, suffix: str | None = None) -> str:
    """Cria um run id com timestamp UTC e sufixo curto único."""
    moment = timestamp or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    short = suffix or uuid4().hex[:8]
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{short}"


def hash_file(path: str | Path) -> str:
    """Retorna o hash SHA256 de um arquivo."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(root: str | Path | None = None) -> str | None:
    """Retorna o commit Git atual quando disponível."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root or "."),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def package_versions(packages: list[str]) -> dict[str, str | None]:
    """Coleta versões instaladas de bibliotecas importantes."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_info() -> dict[str, Any]:
    """Retorna informações de plataforma, Python e disponibilidade de CPU/GPU."""
    gpu: dict[str, Any] = {"tensorflow": [], "torch_cuda_available": None}
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        gpu["tensorflow"] = [device.name for device in tf.config.list_physical_devices("GPU")]
    else:
        gpu["tensorflow"] = "not_loaded"
    if "torch" in sys.modules:
        import torch

        gpu["torch_cuda_available"] = bool(torch.cuda.is_available())
        gpu["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    else:
        gpu["torch_cuda_available"] = "not_loaded"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "gpu": gpu,
        "library_versions": package_versions(
            ["pandas", "numpy", "scipy", "pyarrow", "openpyxl", "tensorflow", "ctgan", "torch"]
        ),
    }


def build_manifest(
    run_id: str,
    model: str,
    seed: int,
    requested_rows: int,
    generated_rows: int,
    status: str,
    config: ConfigDict,
    artifact_paths: dict[str, Path],
    started_at_utc: datetime,
    ended_at_utc: datetime,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Cria um dicionário de manifesto para uma execução concluída."""
    hashes = {
        key: hash_file(path)
        for key, path in artifact_paths.items()
        if path.exists() and path.is_file()
    }
    return {
        "run_id": run_id,
        "timestamp_utc": started_at_utc.astimezone(timezone.utc).isoformat(),
        "ended_at_utc": ended_at_utc.astimezone(timezone.utc).isoformat(),
        "duration_seconds": float((ended_at_utc - started_at_utc).total_seconds()),
        "model": model,
        "seed": int(seed),
        "requested_rows": int(requested_rows),
        "generated_rows": int(generated_rows),
        "status": status,
        "config_hash": config_hash(config),
        "artifact_hashes": hashes,
        "git_commit": get_git_commit(root),
        "environment": environment_info(),
    }


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Grava um arquivo JSON com formatação determinística."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str, sort_keys=True)
    return output_path
