from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.generators.identifiers import (
    gerar_cnh,
    gerar_cpf,
    gerar_rg,
    gerar_telefone,
    gerar_titulo_eleitor,
)
from synthetic_br_profiles_gan.validators.brazilian import (
    avaliar_regras_final,
    checar_unicidade,
    validar_cpf,
    validar_formato_rg,
    validar_telefone,
)


class BrazilianValidatorsTest(unittest.TestCase):
    def test_known_cpf_validation(self) -> None:
        self.assertTrue(validar_cpf("529.982.247-25"))
        self.assertFalse(validar_cpf("111.111.111-11"))
        self.assertFalse(validar_cpf("529.982.247-24"))

    def test_generated_cpf_is_valid(self) -> None:
        random.seed(41)
        for _ in range(25):
            self.assertTrue(validar_cpf(gerar_cpf()))

    def test_generated_rg_and_phone_formats(self) -> None:
        random.seed(41)
        self.assertTrue(validar_formato_rg(gerar_rg()))
        self.assertTrue(validar_telefone(gerar_telefone()))

    def test_unicity_counter(self) -> None:
        df = pd.DataFrame({"CPF": ["529.982.247-25", "529.982.247-25", "123.456.789-09"]})
        self.assertEqual(checar_unicidade(df, ["CPF"]), {"CPF": 1})

    def test_final_rules_for_generated_fields(self) -> None:
        random.seed(41)
        df = pd.DataFrame(
            [
                {
                    "Nome": "Pessoa Sintetica",
                    "Gênero": "Feminino",
                    "Data_Nascimento": "01/01/1990",
                    "CPF": gerar_cpf(),
                    "CNH": gerar_cnh(),
                    "RG": gerar_rg(),
                    "Titulo_Eleitor": gerar_titulo_eleitor(),
                    "Telefone": gerar_telefone(),
                    "Renda": 2500.0,
                }
                for _ in range(5)
            ]
        )
        metricas = avaliar_regras_final(df)
        self.assertEqual(metricas["total_erros"], 0)


if __name__ == "__main__":
    unittest.main()

