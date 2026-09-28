"""Compatibilidade temporária para o histórico de execuções da UI."""

from synthetic_br_profiles_gan.services.execution_history import (
    HistoryRecord,
    filter_history,
    history_as_rows,
    history_summary,
    load_history,
    public_history_row,
)

__all__ = [
    "HistoryRecord",
    "filter_history",
    "history_as_rows",
    "history_summary",
    "load_history",
    "public_history_row",
]
