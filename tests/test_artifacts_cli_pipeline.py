from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.cli import main
from synthetic_br_profiles_gan.config import deep_merge, load_yaml_config, save_yaml_config
from synthetic_br_profiles_gan.artifacts import export_dataset
from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable
from synthetic_br_profiles_gan.manifest import build_run_id
from synthetic_br_profiles_gan.pipeline import DEFAULT_PIPELINE_CONFIG, run_pipeline


class ArtifactsCliPipelineTest(unittest.TestCase):
    def small_config(self, root: Path) -> dict:
        return deep_merge(
            DEFAULT_PIPELINE_CONFIG,
            {
                "artifacts_root": str(root / "artifacts"),
                "seed": 41,
                "reference_date": "2026-07-26",
                "model": "programmatic",
                "calibration": {"seed": 41, "num_rows": 80, "holdout_fraction": 0.25},
                "models": {"programmatic": {"seed": 999}},
                "generation": {"rows": 12, "batch_size": 20, "max_batches": 3, "date_format": "%Y-%m-%d"},
                "export": {"xlsx": False},
                "quality_gates": {
                    "assessment_mode": "smoke",
                    "total_variation_distance_max": {"value": 1.0, "mandatory": False},
                    "correlation_difference_max": {"value": 1.0, "mandatory": False},
                },
            },
        )

    def test_run_id_format(self) -> None:
        self.assertRegex(build_run_id(), re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$"))

    def test_small_programmatic_pipeline_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pipeline(self.small_config(Path(tmp)), model_name="programmatic")
            self.assertEqual(result["status"], "quarantined")
            self.assertEqual(len(result["dataset"]), 12)
            self.assertTrue(result["paths"]["dataset_parquet"].exists())
            self.assertTrue(result["paths"]["manifest"].exists())
            self.assertEqual(result["generation"]["accepted_but_not_selected"], 8)
            saved_config = load_yaml_config(result["paths"]["config"])
            self.assertEqual(saved_config["models"]["programmatic"]["seed"], 999)
            self.assertEqual(saved_config["models"]["programmatic"]["num_rows"], 80)
            self.assertIn("income", saved_config["models"]["programmatic"])

    def test_pipeline_can_approve_when_minimum_sample_gate_is_configured_for_small_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.small_config(Path(tmp))
            config["quality_gates"]["min_evaluation_rows"] = 1
            result = run_pipeline(config, model_name="programmatic")
            self.assertEqual(result["status"], "approved")
            self.assertIn("approved", str(result["paths"]["dataset_parquet"]))

    def test_pipeline_rejected_does_not_write_dataset_to_approved_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.small_config(Path(tmp))
            config["quality_gates"]["invalid_rows_max"] = {"value": -1, "mandatory": True}
            result = run_pipeline(config, model_name="programmatic")
            self.assertEqual(result["status"], "rejected")
            self.assertIn("quarantine", str(result["paths"]["dataset_parquet"]))
            self.assertFalse((Path(config["artifacts_root"]) / "runs" / result["run_id"] / "approved" / "dataset.parquet").exists())

    def test_cli_create_calibration_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config = self.small_config(root)
            save_yaml_config(config, config_path)
            calibration_output = root / "calibration"
            self.assertEqual(
                main(["--log-level", "ERROR", "create-calibration", "--config", str(config_path), "--output", str(calibration_output)]),
                0,
            )
            self.assertTrue((calibration_output / "train.parquet").exists())

            dataset = pd.DataFrame(
                [
                    {
                        "Nome": "Pessoa Sintetica",
                        "Genero": "Feminino",
                        "Data_Nascimento": "1990-01-01",
                        "Idade": 36,
                        "Regiao": "Sudeste",
                        "Estado": "SP",
                        "Municipio": "São Paulo",
                        "DDD": 11,
                        "Telefone": "(11) 90000-1234",
                        "Escolaridade": "Ensino Médio",
                        "Estado_Civil": "Solteiro",
                        "Ocupacao": "Técnico",
                        "Renda": 2500.0,
                        "Dependentes": 0,
                        "CPF": "111.111.111-11",
                        "CNH": "00000000000",
                        "RG": "12.345.678-9",
                        "Titulo_Eleitor": "0000 0000 01 00",
                    }
                ]
            )
            path = root / "dataset.parquet"
            dataset.to_parquet(path, index=False)
            self.assertEqual(
                main(["--log-level", "ERROR", "validate", "--input", str(path), "--config", str(config_path)]),
                2,
            )

    def test_cli_returns_nonzero_for_invalid_config_missing_dependency_and_corrupt_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid_config_path = root / "bad.yaml"
            save_yaml_config({"seed": "not-int", "model": "programmatic"}, invalid_config_path)
            self.assertEqual(main(["--log-level", "ERROR", "pipeline", "--config", str(invalid_config_path)]), 2)

            config_path = root / "config.yaml"
            save_yaml_config(self.small_config(root), config_path)
            with patch(
                "synthetic_br_profiles_gan.models.ctgan.CTGANSynthesizer._ctgan_class",
                side_effect=ModelBackendUnavailable("Install with: pip install -e \".[ctgan]\""),
            ):
                self.assertEqual(
                    main(["--log-level", "ERROR", "train", "--model", "ctgan", "--config", str(config_path)]),
                    2,
                )

            corrupt_dir = root / "corrupt"
            corrupt_dir.mkdir()
            (corrupt_dir / "config.json").write_text("{bad", encoding="utf-8")
            self.assertEqual(
                main(["--log-level", "ERROR", "generate", "--model", "programmatic", "--model-path", str(corrupt_dir), "--rows", "1"]),
                2,
            )

    def test_cli_pipeline_exit_codes_for_approved_quarantined_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            approved_config = self.small_config(root / "approved")
            approved_config["quality_gates"]["min_evaluation_rows"] = 1
            approved_path = root / "approved.yaml"
            save_yaml_config(approved_config, approved_path)
            self.assertEqual(
                main(["--log-level", "ERROR", "pipeline", "--model", "programmatic", "--config", str(approved_path), "--require-approved"]),
                0,
            )

            quarantined_config = self.small_config(root / "quarantined")
            quarantined_path = root / "quarantined.yaml"
            save_yaml_config(quarantined_config, quarantined_path)
            self.assertEqual(
                main(["--log-level", "ERROR", "pipeline", "--model", "programmatic", "--config", str(quarantined_path), "--require-approved"]),
                2,
            )

            rejected_config = self.small_config(root / "rejected")
            rejected_config["quality_gates"]["invalid_rows_max"] = {"value": -1, "mandatory": True}
            rejected_path = root / "rejected.yaml"
            save_yaml_config(rejected_config, rejected_path)
            self.assertEqual(
                main(["--log-level", "ERROR", "pipeline", "--model", "programmatic", "--config", str(rejected_path), "--require-approved"]),
                2,
            )

    def test_export_failure_surfaces_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("parquet failed")):
                with self.assertRaises(OSError):
                    export_dataset(pd.DataFrame([{"a": 1}]), Path(tmp), export_xlsx=False)


if __name__ == "__main__":
    unittest.main()
