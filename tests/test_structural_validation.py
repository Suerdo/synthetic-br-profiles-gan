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
from synthetic_br_profiles_gan.generators.demographics import criar_faker, finalizar_perfis_sinteticos
from synthetic_br_profiles_gan.generators.identifiers import gerar_cnh, gerar_titulo_eleitor
from synthetic_br_profiles_gan.validators.brazilian import validar_cnh, validar_titulo_eleitor
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


class StructuralValidationTest(unittest.TestCase):
    def test_cnh_and_titulo_generators_match_validators(self) -> None:
        rng = random.Random(41)
        for _ in range(20):
            self.assertTrue(validar_cnh(gerar_cnh(rng)))
            self.assertTrue(validar_titulo_eleitor(gerar_titulo_eleitor(rng)))

    def test_location_and_phone_incompatibilities_are_detected(self) -> None:
        core = generate_calibration_dataset(5, seed=41)
        final = finalizar_perfis_sinteticos(
            core,
            fake=criar_faker(41),
            referencia=pd.Timestamp("2026-07-26").to_pydatetime(),
            rng=random.Random(41),
        )
        broken = final.copy()
        broken.loc[0, "Estado"] = "SP"
        broken.loc[0, "Regiao"] = "Norte"
        broken.loc[1, "DDD"] = 11 if broken.loc[1, "Estado"] != "SP" else 81
        report = validate_profile_dataframe(broken, reference_date="2026-07-26").report
        self.assertGreaterEqual(report["invalid_rows"], 2)
        self.assertIn("estado_regiao_incompativel", report["reason_counts"])
        self.assertIn("ddd_estado_incompativel", report["reason_counts"])

    def test_birth_date_mismatch_is_detected(self) -> None:
        core = generate_calibration_dataset(3, seed=5)
        final = finalizar_perfis_sinteticos(
            core,
            fake=criar_faker(5),
            referencia=pd.Timestamp("2026-07-26").to_pydatetime(),
            rng=random.Random(5),
        )
        final.loc[0, "Data_Nascimento"] = "2000-01-01"
        report = validate_profile_dataframe(final, reference_date="2026-07-26").report
        self.assertIn("idade_data_nascimento_incompativel", report["reason_counts"])


if __name__ == "__main__":
    unittest.main()
