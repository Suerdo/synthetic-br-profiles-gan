from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.diagnostics.raw_validity import (
    failure_intersections,
    metric_semantics,
    missing_occupation_diagnostic,
    postprocessing_field_changes,
    raw_rule_masks,
    wasserstein_income_diagnostic,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Idade": 30,
                "Genero": "Feminino",
                "Regiao": "Sudeste",
                "Estado": "SP",
                "Municipio": "São Paulo",
                "Escolaridade": "Ensino Médio",
                "Estado_Civil": "Solteiro",
                "Ocupacao": "Atendente",
                "Renda": 2500.0,
                "Dependentes": 0,
                "DDD": 11,
            },
            {
                "Idade": 19,
                "Genero": "Masculino",
                "Regiao": "Norte",
                "Estado": "SP",
                "Municipio": "Manaus",
                "Escolaridade": "Fundamental",
                "Estado_Civil": "Solteiro",
                "Ocupacao": "Médico",
                "Renda": 700.0,
                "Dependentes": 7,
                "DDD": 92,
            },
            {
                "Idade": 22,
                "Genero": "Outro",
                "Regiao": "Sudeste",
                "Estado": "SP",
                "Municipio": "Campinas",
                "Escolaridade": "Superior Completo",
                "Estado_Civil": "Viúvo",
                "Ocupacao": "Engenheiro",
                "Renda": 6000.0,
                "Dependentes": 1,
                "DDD": 19,
            },
        ]
    )


class RawValidityDiagnosticsTest(unittest.TestCase):
    def test_metric_semantics_declares_denominators_and_units(self) -> None:
        semantics = metric_semantics()
        self.assertEqual(semantics["raw_structural_validity_rate"]["denominator"], "linhas selecionadas para o dataset final")
        self.assertEqual(semantics["postprocessing_rejection_rate"]["population"], "final_candidates")
        self.assertEqual(semantics["candidate_acceptance_rate"]["unit"], "taxa entre 0 e 1")

    def test_rule_masks_cover_geographic_professional_and_non_relational_validity(self) -> None:
        masks = raw_rule_masks(_frame())
        self.assertEqual(float(masks["geographic_joint"].mean()), 2 / 3)
        self.assertEqual(float(masks["professional_joint"].mean()), 1 / 3)
        self.assertEqual(float(masks["non_relational_joint"].mean()), 2 / 3)
        self.assertFalse(bool(masks["occupation_education"].iloc[1]))
        self.assertFalse(bool(masks["marital_age"].iloc[2]))

    def test_failure_intersections_and_cooccurrence(self) -> None:
        failure_frame = pd.DataFrame(
            {
                "region_state": [False, True, False],
                "state_ddd": [False, True, False],
                "occupation_education": [False, True, True],
            }
        )
        result = failure_intersections(failure_frame)
        self.assertEqual(result["failure_count_distribution"]["one_failure"], 1)
        self.assertEqual(result["failure_count_distribution"]["three_or_more_failures"], 1)
        pairs = {(row["rule_a"], row["rule_b"]): row["count"] for row in result["cooccurrence_rows"]}
        self.assertEqual(pairs[("region_state", "state_ddd")], 1)
        self.assertEqual(pairs[("state_ddd", "occupation_education")], 1)

    def test_postprocessing_field_changes_separates_repair_replacement_and_rejection(self) -> None:
        raw = _frame()
        final = raw.copy()
        final.loc[1, "Regiao"] = "Sudeste"
        final.loc[1, "Municipio"] = "São Paulo"
        final.loc[1, "DDD"] = 11
        final.loc[2, "Idade"] = 25
        result = postprocessing_field_changes(raw, final, selected_indices=[0, 1])
        classification = result["classification"]
        self.assertEqual(classification["repaired_rows"], 1)
        self.assertEqual(classification["replaced_rows"], 0)
        self.assertEqual(classification["rejected_rows"], 1)
        self.assertEqual(classification["unchanged_rows"], 1)
        self.assertEqual(classification["multiple_field_change_rows"], 1)
        fields = {row["field"] for row in result["field_summaries"]}
        self.assertFalse({"CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"} & fields)

    def test_wasserstein_reports_absolute_brl_and_normalized_values(self) -> None:
        holdout = pd.DataFrame({"Renda": [1000.0, 2000.0, 3000.0, 4000.0]})
        raw = pd.DataFrame({"Renda": [1000.0, 2000.0, 5000.0, 6000.0]})
        final = pd.DataFrame({"Renda": [1000.0, 2000.0, 3000.0, 4500.0]})
        result = wasserstein_income_diagnostic(holdout, raw, final)
        self.assertEqual(result["wasserstein_distance_absolute_brl"]["unit"], "BRL")
        self.assertGreater(result["wasserstein_distance_absolute_brl"]["raw"], result["wasserstein_distance_absolute_brl"]["final"])
        self.assertEqual(result["wasserstein_distance_normalized"]["unit"], "reference_iqr_fallback_std")

    def test_missing_occupation_reports_rejected_candidates_without_sensitive_values(self) -> None:
        train = pd.DataFrame({"Ocupacao": ["Médico", "Atendente"]})
        holdout = pd.DataFrame({"Ocupacao": ["Médico"]})
        raw = pd.DataFrame({"Ocupacao": ["Atendente", "Médico", "Médico"]})
        final = pd.DataFrame({"Ocupacao": ["Atendente", "Médico", "Atendente"]})
        result = missing_occupation_diagnostic("Médico", train, holdout, raw, final, selected_indices=[0])
        self.assertEqual(result["train"]["count"], 1)
        self.assertEqual(result["final_selected"]["count"], 0)
        self.assertTrue(result["appeared_in_rejected_or_surplus_candidates"])
        self.assertNotIn("CPF", str(result))


if __name__ == "__main__":
    unittest.main()
