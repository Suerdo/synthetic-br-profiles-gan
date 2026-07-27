from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.generation import select_valid_candidates


class GenerationAccountingTest(unittest.TestCase):
    def assert_invariants(self, accounting: dict) -> None:
        self.assertEqual(accounting["total_candidates"], accounting["accepted_by_rules"] + accounting["rejected_by_rules"])
        self.assertEqual(accounting["accepted_by_rules"], accounting["selected"] + accounting["accepted_but_not_selected"])
        expected_rate = 0.0 if accounting["total_candidates"] == 0 else accounting["accepted_by_rules"] / accounting["total_candidates"]
        self.assertEqual(accounting["real_acceptance_rate"], expected_rate)

    def test_zero_candidates_accounting(self) -> None:
        result = select_valid_candidates(pd.DataFrame(), pd.Series(dtype=bool), n_target=10)
        expected = {
            "total_candidates": 0,
            "accepted_by_rules": 0,
            "selected": 0,
            "accepted_but_not_selected": 0,
            "rejected_by_rules": 0,
        }
        for key, value in expected.items():
            self.assertEqual(result.accounting[key], value)
        self.assert_invariants(result.accounting)

    def test_surplus_valid_rows_in_last_batch_accounting(self) -> None:
        candidates = pd.DataFrame({"value": range(5)})
        result = select_valid_candidates(candidates, pd.Series([True, True, True, True, False]), n_target=3)
        self.assertEqual(result.accounting["accepted_but_not_selected"], 1)
        self.assert_invariants(result.accounting)


if __name__ == "__main__":
    unittest.main()
