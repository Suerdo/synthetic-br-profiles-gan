"""Candidate selection and generation accounting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Selected candidates plus accounting metrics."""

    selected: pd.DataFrame
    accounting: dict[str, Any]


def select_valid_candidates(
    candidates: pd.DataFrame,
    valid_mask: pd.Series,
    n_target: int,
    rejection_reasons: dict[str, int] | None = None,
    attempts: int = 1,
) -> CandidateSelectionResult:
    """Select up to ``n_target`` valid rows while accounting for surplus valid rows."""
    mask = valid_mask.reindex(candidates.index).fillna(False).astype(bool)
    accepted = candidates.loc[mask].reset_index(drop=True)
    selected = accepted.iloc[:n_target].copy()
    accepted_by_rules = int(len(accepted))
    selected_count = int(len(selected))
    accounting = {
        "total_candidates": int(len(candidates)),
        "accepted_by_rules": accepted_by_rules,
        "selected": selected_count,
        "accepted_but_not_selected": int(max(accepted_by_rules - selected_count, 0)),
        "rejected_by_rules": int(len(candidates) - accepted_by_rules),
        "attempts_or_batches": int(attempts),
        "real_acceptance_rate": float(0.0 if len(candidates) == 0 else accepted_by_rules / len(candidates)),
        "rejection_reasons": rejection_reasons or {},
    }
    return CandidateSelectionResult(selected=selected, accounting=accounting)
