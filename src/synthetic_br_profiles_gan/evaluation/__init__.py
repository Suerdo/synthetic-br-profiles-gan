"""Utilitários de avaliação e quality gates independentes de modelo."""

from synthetic_br_profiles_gan.evaluation.metrics import evaluate_synthetic_data
from synthetic_br_profiles_gan.evaluation.privacy import duplicate_base_row_metrics, exact_match_metrics, privacy_metrics
from synthetic_br_profiles_gan.evaluation.quality_gates import QualityGateResult, evaluate_quality_gates

__all__ = [
    "QualityGateResult",
    "evaluate_quality_gates",
    "evaluate_synthetic_data",
    "duplicate_base_row_metrics",
    "exact_match_metrics",
    "privacy_metrics",
]
