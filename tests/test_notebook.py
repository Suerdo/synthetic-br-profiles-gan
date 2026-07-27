from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class NotebookTest(unittest.TestCase):
    def test_notebook_imports_package_without_defining_pipeline_functions(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "geracao_de_dados_pessoais_sinteticos_lgpd.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
        self.assertIn("from synthetic_br_profiles_gan.pipeline import", code)
        self.assertNotIn("def gerar_", code)
        self.assertNotIn("class DataPreprocessor", code)
        self.assertNotIn("def train_gan", code)


if __name__ == "__main__":
    unittest.main()
