from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.calibration import _sample_income
from synthetic_br_profiles_gan.evaluation.income_calibration import (
    REQUIRED_INCOME_OCCUPATIONS,
    distribution_overlap_coefficient,
    quantile_overlap,
    run_income_calibration_analysis,
)
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, INCOME_MODEL_VERSION
from synthetic_br_profiles_gan.models.profiles import ctgan_income_v3_recommended_candidate_profile
from synthetic_br_profiles_gan.ui.model_catalog import model_catalog_by_name


class IncomeV3RefinementTest(unittest.TestCase):
    def _incomes(self, version: int, variant: str, seed: int, occupation: str = "Mecânico") -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                _sample_income(
                    rng,
                    age=48,
                    education="Ensino Médio",
                    occupation=occupation,
                    region="Sudeste",
                    minimum=800.0,
                    maximum=50000.0,
                    income_model_version=version,
                    income_model_variant=variant,
                )
                for _ in range(1200)
            ],
            dtype=float,
        )

    def test_income_model_version_three_is_explicit(self) -> None:
        self.assertEqual(CATEGORICAL_VOCABULARY_VERSION, 2)
        self.assertEqual(INCOME_MODEL_VERSION, 3)

    def test_income_v3_is_reproducible_bounded_and_gender_independent(self) -> None:
        self.assertNotIn("genero", inspect.signature(_sample_income).parameters)
        self.assertNotIn("gender", inspect.signature(_sample_income).parameters)

        first = self._incomes(3, "selected", 41)
        second = self._incomes(3, "selected", 41)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(first.min(), 800.0)
        self.assertLessEqual(first.max(), 50000.0)
        self.assertGreater(first.max(), 10000.0)
        self.assertLess(float((first > 10000.0).mean()), 0.05)

    def test_income_v3_reduces_v1_tail_without_matching_v2_compression(self) -> None:
        v1 = self._incomes(1, "historical", 42)
        v2 = self._incomes(2, "current", 42)
        v3 = self._incomes(3, "selected", 42)

        self.assertLess(np.quantile(v3, 0.99), np.quantile(v1, 0.99))
        self.assertGreater(np.std(v3, ddof=1), np.std(v2, ddof=1))
        self.assertGreater(distribution_overlap_coefficient(v3, v2), 0.80)
        self.assertGreater(quantile_overlap(v3, v2), 0.60)

    def test_income_calibration_analysis_separates_versions_and_selects_v3(self) -> None:
        analysis = run_income_calibration_analysis(
            seeds=[41],
            rows_per_occupation=80,
            occupations=("Mecânico", "Atendente", "Engenheiro"),
        )
        versions = {row["version_name"] for row in analysis["rows"]}
        self.assertEqual(versions, {"income_v1", "income_v2", "income_v3_candidate_a", "income_v3_candidate_b"})
        self.assertEqual(analysis["selected_calibration"]["income_model_version"], 3)
        self.assertIn(analysis["selected_calibration"]["classification"], {"selected_calibration", "candidate"})
        self.assertTrue(analysis["compression"])
        self.assertTrue(analysis["overlap"])
        self.assertTrue(analysis["ranking"])

    def test_required_income_occupations_are_the_controlled_set(self) -> None:
        self.assertEqual(
            set(REQUIRED_INCOME_OCCUPATIONS),
            {
                "Mecânico",
                "Eletricista",
                "Pedreiro",
                "Motorista",
                "Técnico de Informática",
                "Técnico de Enfermagem",
                "Serviços Gerais",
                "Atendente",
                "Vendedor",
                "Professor",
                "Engenheiro",
                "Médico",
                "Autônomo",
                "Microempreendedor",
            },
        )

    def test_ctgan_candidate_c_profile_is_frozen_without_seed_or_auto_promotion(self) -> None:
        profile = ctgan_income_v3_recommended_candidate_profile()
        ctgan = profile["ctgan"]
        self.assertEqual(profile["profile_name"], "ctgan_income_v3_recommended_candidate")
        self.assertEqual(profile["purpose"], "recommended_candidate")
        self.assertNotIn(profile["purpose"], {"approved", "default", "production"})
        self.assertEqual(profile["categorical_vocabulary_version"], 2)
        self.assertEqual(profile["income_model_version"], 3)
        self.assertNotIn("seed", ctgan)
        self.assertEqual(ctgan["epochs"], 20)
        self.assertEqual(ctgan["batch_size"], 500)
        self.assertEqual(ctgan["generator_lr"], 0.0001)
        self.assertEqual(ctgan["discriminator_lr"], 0.0001)
        self.assertEqual(ctgan["pac"], 10)

    def test_three_model_roles_remain_registered(self) -> None:
        catalog = model_catalog_by_name()
        self.assertEqual(set(catalog), {"programmatic", "ctgan", "simple_gan"})
        self.assertTrue(catalog["programmatic"].recommended)
        self.assertFalse(catalog["ctgan"].recommended)
        self.assertTrue(catalog["simple_gan"].experimental)


if __name__ == "__main__":
    unittest.main()
