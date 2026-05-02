from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.reports.execution import exportar_resultados


class ReportsTest(unittest.TestCase):
    def test_export_report_and_dataset(self) -> None:
        dataset = pd.DataFrame([{"Nome": "Pessoa Sintetica", "Renda": 1234.56}])
        relatorio = {"seed": 41, "n_target": 1}
        tmpdir = ROOT / "tests" / ".tmp" / "report_export"

        if tmpdir.exists():
            shutil.rmtree(tmpdir)

        try:
            paths = exportar_resultados(
                dataset,
                relatorio,
                tmpdir,
                dataset_filename="dados.csv",
                report_filename="relatorio_execucao.json",
            )

            self.assertTrue(paths["dataset"].exists())
            self.assertTrue(paths["relatorio"].exists())

            with paths["relatorio"].open(encoding="utf-8") as file:
                saved = json.load(file)

            self.assertEqual(saved["seed"], 41)
            self.assertEqual(saved["n_target"], 1)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
