from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.generation import select_valid_candidates
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.pipeline import generate_profiles
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


class FixedBatchSynthesizer:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def sample(self, num_rows: int) -> pd.DataFrame:
        repeats = (int(num_rows) // len(self.frame)) + 1
        return pd.concat([self.frame] * repeats, ignore_index=True).iloc[: int(num_rows)].copy()


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

    def test_global_mask_overrides_batch_mask(self) -> None:
        candidates = pd.DataFrame({"value": [1, 2, 3]})
        batch_mask = pd.Series([True, True, True])
        global_mask = pd.Series([True, False, True])

        result = select_valid_candidates(
            candidates,
            global_mask,
            n_target=2,
            batch_valid_mask=batch_mask,
        )

        self.assertEqual(result.selected["value"].tolist(), [1, 3])
        self.assertEqual(result.accounting["accepted_by_batch_rules"], 3)
        self.assertEqual(result.accounting["accepted_by_global_rules"], 2)
        self.assertEqual(result.accounting["rejected_by_global_rules"], 1)
        self.assert_invariants(result.accounting)

    def test_generate_profiles_preserves_identifier_uniqueness_across_batches(self) -> None:
        core = pd.DataFrame(
            {
                "Idade": [30],
                "Genero": ["Feminino"],
                "Regiao": ["Sudeste"],
                "Estado": ["SP"],
                "Municipio": ["São Paulo"],
                "Escolaridade": ["Ensino Médio"],
                "Estado_Civil": ["Solteiro"],
                "Ocupacao": ["Atendente"],
                "Renda": [2500.0],
                "Dependentes": [0],
                "DDD": [11],
            }
        )
        cpf_values = iter(["291.417.776-38", "398.259.791-94", "291.417.776-38", "011.524.493-03", "341.672.110-17"])
        with patch("synthetic_br_profiles_gan.generators.demographics.gerar_cpf", side_effect=lambda rng: next(cpf_values)):
            dataset, accounting, validation = generate_profiles(
                FixedBatchSynthesizer(core),
                n_target=3,
                metadata=default_metadata(),
                seed=43,
                reference_date="2026-07-26",
                batch_size=2,
                max_batches=3,
                date_format="%Y-%m-%d",
            )

        self.assertEqual(len(dataset), 3)
        for column in ["CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"]:
            self.assertEqual(int(dataset[column].duplicated().sum()), 0)
        self.assertEqual(validation["invalid_rows"], 0)
        self.assertEqual(accounting["selected"], 3)
        self.assertEqual(accounting["accepted_by_batch_rules"], 4)
        self.assertEqual(accounting["accepted_by_global_rules"], 4)
        self.assertEqual(accounting["rejected_by_global_rules"], 0)
        self.assertEqual(accounting["cross_batch_identifier_duplicates"], 0)

    def test_seed43_programmatic_multiple_batches_passes_global_validation(self) -> None:
        from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer

        metadata = default_metadata()
        dataset, accounting, validation = generate_profiles(
            ProgrammaticSynthesizer({"seed": 100046}),
            n_target=300,
            metadata=metadata,
            seed=43,
            reference_date="2026-07-26",
            batch_size=128,
            max_batches=5,
            date_format="%Y-%m-%d",
        )
        final_validation = validate_profile_dataframe(dataset, metadata=metadata, reference_date="2026-07-26").report

        self.assertEqual(len(dataset), 300)
        self.assertEqual(accounting["selected"], 300)
        self.assertEqual(validation["invalid_rows"], 0)
        self.assertEqual(final_validation["invalid_rows"], 0)


if __name__ == "__main__":
    unittest.main()
