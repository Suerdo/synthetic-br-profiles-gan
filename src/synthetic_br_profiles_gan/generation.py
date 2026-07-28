"""Utilitários de seleção de candidatos e contabilização da geração."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Candidatos selecionados e métricas de contabilização."""

    selected: pd.DataFrame
    accounting: dict[str, Any]


def select_valid_candidates(
    candidates: pd.DataFrame,
    valid_mask: pd.Series,
    n_target: int,
    rejection_reasons: dict[str, int] | None = None,
    attempts: int = 1,
    batch_valid_mask: pd.Series | None = None,
) -> CandidateSelectionResult:
    """Seleciona até ``n_target`` linhas válidas e contabiliza excedentes válidos."""
    mask = valid_mask.reindex(candidates.index).fillna(False).astype(bool)
    if batch_valid_mask is None:
        batch_mask = mask
    else:
        batch_mask = batch_valid_mask.reindex(candidates.index).fillna(False).astype(bool)
    accepted = candidates.loc[mask].reset_index(drop=True)
    selected = accepted.iloc[:n_target].copy()
    accepted_by_global_rules = int(len(accepted))
    accepted_by_batch_rules = int(batch_mask.sum())
    selected_count = int(len(selected))
    accounting = {
        "total_candidates": int(len(candidates)),
        "accepted_by_rules": accepted_by_global_rules,
        "accepted_by_batch_rules": accepted_by_batch_rules,
        "accepted_by_global_rules": accepted_by_global_rules,
        "selected": selected_count,
        "accepted_but_not_selected": int(max(accepted_by_global_rules - selected_count, 0)),
        "rejected_by_rules": int(len(candidates) - accepted_by_global_rules),
        "rejected_by_batch_rules": int(len(candidates) - accepted_by_batch_rules),
        "rejected_by_global_rules": int(len(candidates) - accepted_by_global_rules),
        "attempts_or_batches": int(attempts),
        "real_acceptance_rate": float(0.0 if len(candidates) == 0 else accepted_by_global_rules / len(candidates)),
        "batch_acceptance_rate": float(0.0 if len(candidates) == 0 else accepted_by_batch_rules / len(candidates)),
        "global_acceptance_rate": float(0.0 if len(candidates) == 0 else accepted_by_global_rules / len(candidates)),
        "rejection_reasons": rejection_reasons or {},
    }
    return CandidateSelectionResult(selected=selected, accounting=accounting)
