from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset, split_train_holdout
from synthetic_br_profiles_gan.evaluation.metrics import (
    categorical_column_metrics,
    correlation_metrics,
    evaluate_synthetic_data,
    numeric_column_metrics,
)
from synthetic_br_profiles_gan.evaluation.income import conditional_income_report
from synthetic_br_profiles_gan.evaluation.privacy import duplicate_base_row_metrics, exact_match_metrics
from synthetic_br_profiles_gan.evaluation.quality_gates import evaluate_quality_gates
from synthetic_br_profiles_gan.generation import select_valid_candidates


class EvaluationAndGatesTest(unittest.TestCase):
    def test_numeric_and_categorical_metrics(self) -> None:
        numeric = numeric_column_metrics(pd.Series([1, 2, 3, 4]), pd.Series([2, 3, 4, 5]))
        self.assertIn("wasserstein_distance", numeric)
        self.assertIn("wasserstein_distance_normalized", numeric)
        self.assertGreater(numeric["ks_statistic"], 0)

        categorical = categorical_column_metrics(pd.Series(["A", "A", "B"]), pd.Series(["A", "C", "C"]))
        self.assertEqual(categorical["missing_categories"], ["B"])
        self.assertEqual(categorical["unexpected_categories"], ["C"])
        self.assertGreater(categorical["total_variation_distance"], 0)

    def test_metrics_handle_empty_zero_variance_nan_and_inf(self) -> None:
        self.assertEqual(numeric_column_metrics(pd.Series([], dtype=float), pd.Series([1]))["error"], "empty_numeric_series")
        zero_var = numeric_column_metrics(pd.Series([3, 3, 3]), pd.Series([3, np.nan, np.inf]))
        self.assertIsNone(zero_var["wasserstein_distance_normalized"])
        categorical = categorical_column_metrics(pd.Series(["A", None]), pd.Series(["A", "B"]))
        self.assertIn("<NA>", categorical["missing_categories"])
        self.assertIn("B", categorical["unexpected_categories"])

    def test_correlation_and_privacy_metrics_are_reported(self) -> None:
        calibration = generate_calibration_dataset(120, seed=41)
        train, holdout = split_train_holdout(calibration, seed=41)
        synthetic = generate_calibration_dataset(40, seed=999)
        report = evaluate_synthetic_data(synthetic, train, holdout)
        self.assertIn("against_train", report)
        self.assertIn("against_holdout", report)
        self.assertIn("privacy", report)
        corr = correlation_metrics(train, synthetic, ["Idade", "Renda", "Dependentes"])
        self.assertIn("max_abs_difference", corr["summary"])

    def test_candidate_accounting_counts_surplus_valid_rows(self) -> None:
        candidates = pd.DataFrame({"value": range(6)})
        mask = pd.Series([True, True, False, True, True, False])
        result = select_valid_candidates(candidates, mask, n_target=3, attempts=1)
        self.assertEqual(result.accounting["total_candidates"], 6)
        self.assertEqual(result.accounting["accepted_by_rules"], 4)
        self.assertEqual(result.accounting["selected"], 3)
        self.assertEqual(result.accounting["accepted_but_not_selected"], 1)
        self.assertEqual(result.accounting["rejected_by_rules"], 2)

    def test_quality_gate_statuses(self) -> None:
        valid_eval = {
            "privacy": {"exact_train_match_rate": 0.0},
            "against_holdout": {
                "categorical": {"Genero": {"total_variation_distance": 0.0}},
                "correlations": {"summary": {"max_abs_difference": 0.0}},
            },
            "row_counts": {"synthetic": 100, "train": 100, "holdout": 20},
        }
        approved = evaluate_quality_gates({"invalid_rows": 0, "reason_counts": {}}, valid_eval)
        self.assertEqual(approved.status, "approved")

        rejected = evaluate_quality_gates({"invalid_rows": 1, "reason_counts": {}}, valid_eval)
        self.assertEqual(rejected.status, "rejected")

        optional_fail = evaluate_quality_gates(
            {"invalid_rows": 0, "reason_counts": {}},
            {
                "privacy": {"exact_train_match_rate": 0.0},
                "row_counts": {"synthetic": 100, "train": 100, "holdout": 20},
                "against_holdout": {
                    "categorical": {"Genero": {"total_variation_distance": 0.9}},
                    "correlations": {"summary": {"max_abs_difference": 0.0}},
                },
            },
        )
        self.assertEqual(optional_fail.status, "quarantined")

    def test_quality_gate_rejects_missing_or_nan_mandatory_metric(self) -> None:
        missing = evaluate_quality_gates(
            {"invalid_rows": 0, "reason_counts": {}},
            {"against_holdout": {"categorical": {}, "correlations": {"summary": {}}}, "row_counts": {"synthetic": 100}},
        )
        self.assertEqual(missing.status, "rejected")
        self.assertEqual(missing.failures[0]["reason"], "metric_missing_or_invalid")

        nan_metric = evaluate_quality_gates(
            {"invalid_rows": 0, "reason_counts": {}},
            {
                "privacy": {"exact_train_match_rate": float("nan")},
                "against_holdout": {"categorical": {}, "correlations": {"summary": {}}},
                "row_counts": {"synthetic": 100},
            },
        )
        self.assertEqual(nan_metric.status, "rejected")

    def test_duplicate_base_row_metrics_distinguish_occurrences_and_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "Idade": [30, 30, 30, 40, 40, 50],
                "Genero": ["A", "A", "A", "B", "B", "C"],
                "Renda": [1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 3000.0],
            }
        )
        metrics = duplicate_base_row_metrics(frame, ["Idade", "Genero", "Renda"])
        self.assertEqual(metrics["total_rows"], 6)
        self.assertEqual(metrics["unique_rows"], 3)
        self.assertEqual(metrics["duplicated_occurrences"], 3)
        self.assertEqual(metrics["duplicated_groups"], 2)
        self.assertEqual(metrics["rows_in_duplicate_groups"], 5)
        self.assertEqual(metrics["largest_duplicate_group"], 3)
        self.assertEqual(metrics["duplicate_group_size_distribution"], {"2": 1, "3": 1})
        self.assertNotIn("CPF", metrics["columns_used"])

    def test_exact_match_metrics_canonicalize_values_and_reference_duplicates(self) -> None:
        synthetic = pd.DataFrame(
            {
                "Escolaridade": ["Ensino Medio", "Superior Completo", "Pós-graduação"],
                "Renda": [1000.004, 2000.0, 3000.0],
                "Idade": [30.0, 40, 50],
            }
        )
        reference = pd.DataFrame(
            {
                "Escolaridade": ["Ensino Médio", "Ensino Médio", "Superior Completo"],
                "Renda": [1000.0, 1000.0, 2000.0],
                "Idade": [30, 30, 40],
            }
        )
        metrics = exact_match_metrics(synthetic, reference, ["Escolaridade", "Renda", "Idade"])
        self.assertEqual(metrics["exact_match_count"], 2)
        self.assertEqual(metrics["distinct_reference_rows_matched"], 2)
        self.assertEqual(len(metrics["matched_row_hashes"]), 2)
        self.assertTrue(all(len(item) == 64 for item in metrics["matched_row_hashes"]))

    def test_duplicate_base_row_gate_is_informative(self) -> None:
        evaluation = {
            "privacy": {
                "exact_train_match_rate": 0.0,
                "duplicate_base_rows": {"duplicate_row_rate": 0.02},
            },
            "against_holdout": {
                "categorical": {"Genero": {"total_variation_distance": 0.0}},
                "correlations": {"summary": {"max_abs_difference": 0.0}},
            },
            "row_counts": {"synthetic": 100, "train": 100, "holdout": 20},
        }
        result = evaluate_quality_gates({"invalid_rows": 0, "reason_counts": {}}, evaluation)
        self.assertEqual(result.status, "quarantined")
        self.assertEqual(result.failures[0]["gate"], "duplicate_base_row_rate_max")

    def test_conditional_income_report_uses_group_thresholds(self) -> None:
        reference = pd.DataFrame(
            {
                "Ocupacao": ["Mecânico"] * 40 + ["Médico"] * 40,
                "Escolaridade": ["Ensino Médio"] * 40 + ["Superior Completo"] * 40,
                "Idade": [35] * 80,
                "Regiao": ["Sudeste"] * 80,
                "Renda": [3000 + index for index in range(40)] + [10000 + index for index in range(40)],
            }
        )
        synthetic = reference.copy()
        synthetic.loc[0, "Renda"] = 12000.0
        report = conditional_income_report(reference, synthetic, minimum_group_rows=30)
        self.assertGreater(report["summary"]["conditional_groups_compared"], 0)
        self.assertIn("summary_rows", report)
        self.assertIn("comparison_rows", report)
        self.assertIn("tail_events", report)


if __name__ == "__main__":
    unittest.main()
