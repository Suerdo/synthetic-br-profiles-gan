from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset, split_train_holdout
from synthetic_br_profiles_gan.domain.brazil import STATE_DDDS, STATE_MUNICIPALITIES, region_for_state
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS, MODEL_COLUMNS, default_metadata


class MetadataCalibrationTest(unittest.TestCase):
    def test_metadata_declares_schema_and_ctgan_discrete_columns(self) -> None:
        metadata = default_metadata()
        self.assertEqual(metadata.model_columns, MODEL_COLUMNS)
        self.assertEqual(metadata.final_columns, FINAL_COLUMNS)
        self.assertIn("DDD", metadata.categorical_columns(include_discrete_numeric=True))
        self.assertIn("Renda", metadata.numeric_columns())

    def test_calibration_is_reproducible_and_semantically_consistent(self) -> None:
        first = generate_calibration_dataset(200, seed=123)
        second = generate_calibration_dataset(200, seed=123)
        pd.testing.assert_frame_equal(first, second)

        for _, row in first.iterrows():
            self.assertEqual(row["Regiao"], region_for_state(row["Estado"]))
            self.assertIn(row["Municipio"], STATE_MUNICIPALITIES[row["Estado"]])
            self.assertIn(int(row["DDD"]), STATE_DDDS[row["Estado"]])

        self.assertGreater(first["Renda"].mean(), first["Renda"].median())

    def test_split_train_holdout_is_seeded_and_disjoint(self) -> None:
        df = pd.DataFrame({"id": range(20), "value": range(20)})
        train_a, holdout_a = split_train_holdout(df, holdout_fraction=0.25, seed=41)
        train_b, holdout_b = split_train_holdout(df, holdout_fraction=0.25, seed=41)
        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(holdout_a, holdout_b)
        self.assertTrue(set(train_a["id"]).isdisjoint(set(holdout_a["id"])))


if __name__ == "__main__":
    unittest.main()
