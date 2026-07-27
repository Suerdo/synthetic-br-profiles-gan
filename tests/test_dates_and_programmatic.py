from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.generators.demographics import (
    calcular_idade,
    criar_faker,
    finalizar_perfis_sinteticos,
    gerar_data_nascimento_por_idade,
)
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


class DatesAndProgrammaticTest(unittest.TestCase):
    def test_birth_date_matches_age_even_around_leap_years(self) -> None:
        reference = "2024-02-29"
        rng = random.Random(41)
        for age in [18, 19, 40, 65, 85]:
            with self.subTest(age=age):
                birth = gerar_data_nascimento_por_idade(
                    age,
                    referencia=pd.Timestamp(reference).to_pydatetime(),
                    rng=rng,
                    output_format="%Y-%m-%d",
                )
                self.assertEqual(calcular_idade(birth, pd.Timestamp(reference).to_pydatetime()), age)

    def test_programmatic_synthesizer_reproducibility(self) -> None:
        metadata = default_metadata()
        train = generate_calibration_dataset(20, seed=1)
        first = ProgrammaticSynthesizer({"seed": 99})
        second = ProgrammaticSynthesizer({"seed": 99})
        first.fit(train, metadata)
        second.fit(train, metadata)
        pd.testing.assert_frame_equal(first.sample(10), second.sample(10))

    def test_programmatic_final_profiles_pass_structural_validator(self) -> None:
        metadata = default_metadata()
        core = ProgrammaticSynthesizer({"seed": 55}).sample(30)
        final = finalizar_perfis_sinteticos(
            core,
            fake=criar_faker(55),
            referencia=pd.Timestamp("2026-07-26").to_pydatetime(),
            rng=random.Random(55),
        )
        report = validate_profile_dataframe(final, metadata=metadata, reference_date="2026-07-26").report
        self.assertEqual(report["invalid_rows"], 0)
        self.assertTrue(report["is_valid"])


if __name__ == "__main__":
    unittest.main()
