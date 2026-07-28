"""Auditoria sanitizada de eventos operacionais da interface."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

ALLOWED_EVENTS = {
    "session_started",
    "page_viewed",
    "model_selected",
    "generation_requested",
    "generation_succeeded",
    "generation_failed",
    "dataset_download_requested",
    "manifest_download_requested",
}

ALLOWED_FIELDS = {
    "timestamp_utc",
    "event",
    "session_id",
    "page",
    "model",
    "artifact_id",
    "rows",
    "format",
    "seed",
    "column_selection_mode",
    "column_preset",
    "exported_column_count",
    "duration_seconds",
    "status",
    "error_type",
}

FORBIDDEN_FIELD_FRAGMENTS = {
    "cpf",
    "cnh",
    "rg",
    "titulo",
    "telefone",
    "nome",
    "dataset",
    "dataframe",
    "stack",
    "traceback",
    "ip",
    "user_agent",
    "usuario",
    "user",
    "linha",
    "row",
}


@dataclass(frozen=True)
class AuditWriteResult:
    """Resultado da tentativa de escrita de auditoria."""

    written: bool
    error_type: str | None = None


def sanitize_event(event: str, payload: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
    """Normaliza um evento de auditoria removendo campos sensíveis ou não permitidos."""
    event_name = event if event in ALLOWED_EVENTS else "generation_failed"
    sanitized: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
    }
    if session_id:
        sanitized["session_id"] = str(session_id)
    for key, value in (payload or {}).items():
        normalized_key = str(key)
        lower_key = normalized_key.lower()
        if normalized_key not in ALLOWED_FIELDS:
            continue
        if any(fragment in lower_key for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            continue
        sanitized[normalized_key] = _safe_value(value)
    return sanitized


def write_audit_event(
    events_path: str | Path,
    event: str,
    payload: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> AuditWriteResult:
    """Registra um evento em JSONL sem propagar falhas para o usuário."""
    record = sanitize_event(event, payload=payload, session_id=session_id)
    try:
        path = Path(events_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return AuditWriteResult(written=True)
    except OSError as exc:
        LOGGER.warning("ui_audit_write_failed", extra={"error_type": type(exc).__name__})
        return AuditWriteResult(written=False, error_type=type(exc).__name__)


def read_audit_events(events_path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    """Lê os eventos de auditoria mais recentes, ignorando linhas inválidas."""
    path = Path(events_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-int(limit) :]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append({key: record[key] for key in record if key in ALLOWED_FIELDS})
    return records


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return str(type(value).__name__)
