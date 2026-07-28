from __future__ import annotations

import json
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
from synthetic_br_profiles_gan.config import ConfigurationError, load_yaml_config
from synthetic_br_profiles_gan.exceptions import ModelSerializationError
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.models.registry import load_saved_synthesizer
from synthetic_br_profiles_gan.services.generation_service import GenerationRequest, run_generation
from synthetic_br_profiles_gan.services.training_service import TrainingRequest, holdout_rows_for_train_size, run_training


class TrainingGenerationServicesTest(unittest.TestCase):
    def test_programmatic_training_creates_reusable_artifact_without_neural_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "programmatic"
            result = run_training(
                TrainingRequest(
                    model="programmatic",
                    output_path=output,
                    config=load_yaml_config(ROOT / "configs" / "train-programmatic.yaml"),
                    seed=41,
                    train_rows=20,
                    overwrite=False,
                )
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.train_rows, 20)
            self.assertEqual(result.holdout_rows, 5)
            self.assertFalse(manifest["training_required"])
            self.assertEqual(manifest["artifact_type"], "trained_synthesizer")
            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "metadata.json").exists())
            self.assertTrue((output / "training_config.yaml").exists())

    def test_training_refuses_existing_directory_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "model"
            output.mkdir()
            (output / "existing.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                run_training(TrainingRequest("programmatic", output, {}, seed=41, train_rows=10))

    def test_training_rejects_unknown_model_and_exact_holdout(self) -> None:
        self.assertEqual(holdout_rows_for_train_size(4800000, 0.20), 1200000)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigurationError):
                run_training(TrainingRequest("unknown", Path(tmp) / "model", {}, seed=41, train_rows=10))

    def test_simple_gan_and_ctgan_training_manifests_with_mocked_fit(self) -> None:
        def fake_train_synthesizer(model_name, train, metadata, config=None, output_dir=None):
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            metadata.save(output / "metadata.json")
            if model_name == "simple_gan":
                for name in ["generator.keras", "discriminator.keras", "preprocessor.pkl", "config.json", "training_history.json"]:
                    (output / name).write_text("{}", encoding="utf-8")
            elif model_name == "ctgan":
                (output / "model.pkl").write_bytes(b"model")
                (output / "metadata_ctgan.json").write_text("{}", encoding="utf-8")
            return object()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("synthetic_br_profiles_gan.services.training_service.train_synthesizer", side_effect=fake_train_synthesizer):
                simple = run_training(TrainingRequest("simple_gan", root / "simple", {"models": {"simple_gan": {"epochs": 1}}}, train_rows=16))
                ctgan = run_training(TrainingRequest("ctgan", root / "ctgan", {"models": {"ctgan": {"epochs": 1, "batch_size": 10}}}, train_rows=20))
            self.assertTrue(json.loads(simple.manifest_path.read_text(encoding="utf-8"))["training_required"])
            self.assertTrue(json.loads(ctgan.manifest_path.read_text(encoding="utf-8"))["training_required"])
            self.assertTrue((root / "simple" / "training_manifest.json").exists())
            self.assertTrue((root / "ctgan" / "training_manifest.json").exists())

    def test_model_loader_validates_manifest_and_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = run_training(TrainingRequest("programmatic", root / "programmatic", {}, seed=41, train_rows=12))
            loaded = load_saved_synthesizer(trained.output_path)
            self.assertEqual(loaded.model, "programmatic")
            self.assertIsInstance(loaded.synthesizer, ProgrammaticSynthesizer)
            with self.assertRaises(ModelSerializationError):
                load_saved_synthesizer(root / "missing")
            bad = root / "bad"
            bad.mkdir()
            (bad / "training_manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ModelSerializationError):
                load_saved_synthesizer(bad)
            (trained.output_path / "config.json").unlink()
            with self.assertRaises(ModelSerializationError):
                load_saved_synthesizer(trained.output_path)

    def test_loader_uses_expected_model_and_supports_mocked_neural_loaders(self) -> None:
        metadata = default_metadata()
        manifest = {
            "schema_version": 1,
            "artifact_type": "trained_synthesizer",
            "model": "ctgan",
            "model_columns": metadata.model_columns,
            "final_columns": metadata.final_columns,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "model.pkl").write_bytes(b"model")
            metadata.save(root / "metadata.json")
            (root / "metadata_ctgan.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ModelSerializationError):
                load_saved_synthesizer(root, expected_model="programmatic")
            with patch("synthetic_br_profiles_gan.models.registry.CTGANSynthesizer.load", return_value=object()):
                loaded = load_saved_synthesizer(root, expected_model="ctgan")
            self.assertEqual(loaded.model, "ctgan")

    def test_direct_programmatic_generation_exports_csv_json_and_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output_format in ["csv", "json", "parquet"]:
                output = root / f"dataset-{output_format}.{output_format}"
                result = run_generation(
                    GenerationRequest(
                        model="programmatic",
                        model_path=None,
                        num_rows=5,
                        output_path=output,
                        output_format=output_format,
                        seed=41,
                    )
                )
                self.assertEqual(result.num_rows, 5)
                self.assertTrue(output.exists())
                self.assertTrue(result.manifest_path.exists())
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["rows"], 5)
                self.assertIn("não foram consultados", manifest["governance_notice"])
                if output_format == "csv":
                    frame = pd.read_csv(output, sep=";")
                elif output_format == "json":
                    frame = pd.DataFrame(json.loads(output.read_text(encoding="utf-8")))
                else:
                    frame = pd.read_parquet(output)
                self.assertEqual(list(frame.columns), default_metadata().final_columns)
                self.assertEqual(len(frame), 5)

    def test_generation_from_saved_programmatic_model_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = run_training(TrainingRequest("programmatic", root / "model", {}, seed=41, train_rows=12))
            first = root / "first.csv"
            second = root / "second.csv"
            request = {
                "model": None,
                "model_path": trained.output_path,
                "num_rows": 6,
                "output_format": "csv",
                "seed": 123,
            }
            run_generation(GenerationRequest(output_path=first, **request))
            run_generation(GenerationRequest(output_path=second, **request))
            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))

    def test_generation_rejects_bad_rows_format_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "dataset.csv"
            output.write_text("x", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 0, root / "new.csv", "csv"))
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 1, root / "new.csv", "xlsx"))
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 1, output, "csv"))
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("ctgan", None, 1, root / "ctgan.csv", "csv"))

    def test_cli_train_and_generate_programmatic_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "programmatic-smoke"
            dataset = root / "programmatic.csv"
            self.assertEqual(
                main(
                    [
                        "--log-level",
                        "ERROR",
                        "train",
                        "--model",
                        "programmatic",
                        "--config",
                        str(ROOT / "configs" / "train-programmatic.yaml"),
                        "--output",
                        str(model_dir),
                        "--train-rows",
                        "20",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "--log-level",
                        "ERROR",
                        "generate",
                        "--model-path",
                        str(model_dir),
                        "--rows",
                        "5",
                        "--output",
                        str(dataset),
                        "--format",
                        "csv",
                        "--seed",
                        "41",
                    ]
                ),
                0,
            )
            self.assertEqual(len(pd.read_csv(dataset, sep=";")), 5)


if __name__ == "__main__":
    unittest.main()
