from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.column_catalog import (
    COLUMN_CATALOG,
    COLUMN_PRESETS,
    available_column_names,
    resolve_column_selection,
)
from synthetic_br_profiles_gan.config import ConfigurationError
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS


class ColumnCatalogTest(unittest.TestCase):
    def test_catalog_contains_exactly_final_columns_in_canonical_order(self) -> None:
        self.assertEqual(tuple(FINAL_COLUMNS), available_column_names())
        self.assertEqual(len(COLUMN_CATALOG), 18)
        self.assertEqual(len({entry.name for entry in COLUMN_CATALOG}), 18)

    def test_catalog_entries_are_complete_and_dependencies_are_valid(self) -> None:
        valid_names = set(FINAL_COLUMNS)
        for entry in COLUMN_CATALOG:
            self.assertTrue(entry.label)
            self.assertTrue(entry.description)
            self.assertTrue(entry.group)
            self.assertTrue(entry.kind)
            self.assertIn(entry.generated_by, {"model", "postprocessing"})
            self.assertTrue(set(entry.dependencies).issubset(valid_names))

    def test_presets_resolve_in_expected_order(self) -> None:
        self.assertEqual(resolve_column_selection(None, "completo").exported_columns, tuple(FINAL_COLUMNS))
        self.assertEqual(
            resolve_column_selection(None, "demografico").exported_columns,
            (
                "Genero",
                "Data_Nascimento",
                "Idade",
                "Regiao",
                "Estado",
                "Municipio",
                "Escolaridade",
                "Estado_Civil",
                "Ocupacao",
                "Renda",
                "Dependentes",
            ),
        )
        self.assertEqual(
            resolve_column_selection(None, "contato").exported_columns,
            ("Nome", "Regiao", "Estado", "Municipio", "DDD", "Telefone"),
        )
        self.assertEqual(
            resolve_column_selection(None, "documentos").exported_columns,
            ("Nome", "Data_Nascimento", "CPF", "CNH", "RG", "Titulo_Eleitor"),
        )
        self.assertEqual(
            resolve_column_selection(None, "minimo").exported_columns,
            ("Nome", "Idade", "Estado", "CPF"),
        )
        self.assertEqual(set(COLUMN_PRESETS), {"completo", "demografico", "contato", "documentos", "minimo"})

    def test_selection_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "não pode ficar vazia"):
            resolve_column_selection([])
        with self.assertRaisesRegex(ConfigurationError, "Coluna desconhecida: 'Uf'"):
            resolve_column_selection(["Uf"])
        with self.assertRaisesRegex(ConfigurationError, "A coluna 'Nome' foi informada mais de uma vez"):
            resolve_column_selection(["Nome", "CPF", "Nome"])
        with self.assertRaisesRegex(ConfigurationError, "strings"):
            resolve_column_selection(["Nome", 1])  # type: ignore[list-item]
        with self.assertRaisesRegex(ConfigurationError, "Use --columns ou --preset"):
            resolve_column_selection(["Nome"], "minimo")
        with self.assertRaisesRegex(ConfigurationError, "Preset de colunas desconhecido"):
            resolve_column_selection(None, "publico")


if __name__ == "__main__":
    unittest.main()
