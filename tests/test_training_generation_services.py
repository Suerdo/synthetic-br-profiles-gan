from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.cli import main
from synthetic_br_profiles_gan.config import ConfigurationError, load_yaml_config
from synthetic_br_profiles_gan.exceptions import ModelSerializationError, StructuralValidationError
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS, default_metadata
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
                self.assertEqual(manifest["column_selection_mode"], "all")
                self.assertIsNone(manifest["column_preset"])
                self.assertIsNone(manifest["requested_columns"])
                self.assertIn("não foram consultados", manifest["governance_notice"])
                if output_format == "csv":
                    frame = pd.read_csv(output, sep=";")
                elif output_format == "json":
                    frame = pd.DataFrame(json.loads(output.read_text(encoding="utf-8")))
                else:
                    frame = pd.read_parquet(output)
                self.assertEqual(list(frame.columns), default_metadata().final_columns)
                self.assertEqual(len(frame), 5)

    def test_direct_generation_exports_one_selected_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cpf.csv"
            result = run_generation(
                GenerationRequest(
                    model="programmatic",
                    model_path=None,
                    num_rows=4,
                    output_path=output,
                    output_format="csv",
                    seed=41,
                    selected_columns=["CPF"],
                )
            )
            frame = pd.read_csv(output, sep=";")
            self.assertEqual(result.exported_columns, ("CPF",))
            self.assertEqual(list(frame.columns), ["CPF"])
            self.assertEqual(len(frame), 4)

    def test_direct_generation_exports_explicit_selected_columns_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output_format in ["csv", "json", "parquet"]:
                output = root / f"selected-{output_format}.{output_format}"
                result = run_generation(
                    GenerationRequest(
                        model="programmatic",
                        model_path=None,
                        num_rows=5,
                        output_path=output,
                        output_format=output_format,
                        seed=41,
                        selected_columns=["CPF", "Nome", "Estado", "Idade"],
                    )
                )
                self.assertEqual(result.internal_columns, tuple(FINAL_COLUMNS))
                self.assertEqual(result.exported_columns, ("CPF", "Nome", "Estado", "Idade"))
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["column_selection_mode"], "explicit")
                self.assertEqual(manifest["requested_columns"], ["CPF", "Nome", "Estado", "Idade"])
                self.assertEqual(manifest["exported_columns"], ["CPF", "Nome", "Estado", "Idade"])
                self.assertEqual(manifest["internally_generated_columns"], FINAL_COLUMNS)
                self.assertEqual(manifest["validation"]["validated_columns"], FINAL_COLUMNS)
                self.assertTrue(manifest["validation"]["projection_after_validation"])
                if output_format == "csv":
                    frame = pd.read_csv(output, sep=";")
                elif output_format == "json":
                    frame = pd.DataFrame(json.loads(output.read_text(encoding="utf-8")))
                else:
                    frame = pd.read_parquet(output)
                self.assertEqual(list(frame.columns), ["CPF", "Nome", "Estado", "Idade"])
                self.assertEqual(len(frame), 5)

    def test_generation_presets_and_internal_dependencies_do_not_expand_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_generation(
                GenerationRequest(
                    model="programmatic",
                    model_path=None,
                    num_rows=5,
                    output_path=root / "contact.json",
                    output_format="json",
                    seed=41,
                    column_preset="contato",
                )
            )
            frame = pd.DataFrame(json.loads(result.output_path.read_text(encoding="utf-8")))
            self.assertEqual(list(frame.columns), ["Nome", "Regiao", "Estado", "Municipio", "DDD", "Telefone"])
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["column_selection_mode"], "preset")
            self.assertEqual(manifest["column_preset"], "contato")
            self.assertEqual(manifest["internal_dependencies"]["Telefone"], ["Estado", "DDD"])

            explicit = run_generation(
                GenerationRequest(
                    model="programmatic",
                    model_path=None,
                    num_rows=5,
                    output_path=root / "dependency.csv",
                    output_format="csv",
                    seed=41,
                    selected_columns=["Nome", "Telefone", "CPF"],
                )
            )
            dependency_frame = pd.read_csv(explicit.output_path, sep=";")
            self.assertEqual(list(dependency_frame.columns), ["Nome", "Telefone", "CPF"])
            dependency_manifest = json.loads(explicit.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(dependency_manifest["internal_dependencies"]["Nome"], ["Genero"])
            self.assertEqual(dependency_manifest["internal_dependencies"]["Telefone"], ["Estado", "DDD"])
            self.assertNotIn("Estado", dependency_frame.columns)
            self.assertNotIn("DDD", dependency_frame.columns)

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

    def test_generation_from_saved_programmatic_model_supports_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = run_training(TrainingRequest("programmatic", root / "model", {}, seed=41, train_rows=12))
            output = root / "selected.parquet"
            result = run_generation(
                GenerationRequest(
                    model=None,
                    model_path=trained.output_path,
                    num_rows=6,
                    output_path=output,
                    output_format="parquet",
                    seed=123,
                    selected_columns=["Nome", "CPF"],
                )
            )
            self.assertEqual(result.exported_columns, ("Nome", "CPF"))
            self.assertEqual(list(pd.read_parquet(output).columns), ["Nome", "CPF"])

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
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 1, root / "empty.csv", "csv", selected_columns=[]))
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 1, root / "unknown.csv", "csv", selected_columns=["Uf"]))
            with self.assertRaises(ConfigurationError):
                run_generation(
                    GenerationRequest("programmatic", None, 1, root / "duplicate.csv", "csv", selected_columns=["Nome", "Nome"])
                )
            with self.assertRaises(ConfigurationError):
                run_generation(
                    GenerationRequest(
                        "programmatic",
                        None,
                        1,
                        root / "conflict.csv",
                        "csv",
                        selected_columns=["Nome"],
                        column_preset="minimo",
                    )
                )
            with self.assertRaises(ConfigurationError):
                run_generation(GenerationRequest("programmatic", None, 1, root / "preset.csv", "csv", column_preset="desconhecido"))

    def test_validation_runs_on_full_schema_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "selected.csv"
            exported_columns: list[str] = []

            def fake_export(frame: pd.DataFrame, output_path: Path, output_format: str) -> None:
                exported_columns.extend(list(frame.columns))
                output_path.write_text("ok", encoding="utf-8")

            with patch("synthetic_br_profiles_gan.services.generation_service._export_dataset", side_effect=fake_export):
                with patch("synthetic_br_profiles_gan.services.generation_service.validate_profile_dataframe") as validation:
                    validation.return_value = SimpleNamespace(
                        report={
                            "is_valid": True,
                            "invalid_rows": 0,
                            "valid_rows": 3,
                            "n_rows": 3,
                            "reason_counts": {},
                            "details": {"missing_columns": []},
                        }
                    )
                    result = run_generation(
                        GenerationRequest(
                            "programmatic",
                            None,
                            3,
                            output,
                            "csv",
                            seed=41,
                            selected_columns=["Nome", "CPF"],
                        )
                    )
            validated_frame = validation.call_args.args[0]
            self.assertEqual(list(validated_frame.columns), FINAL_COLUMNS)
            self.assertEqual(exported_columns, ["Nome", "CPF"])
            self.assertEqual(result.validation_report["validation_scope"], "full_final_schema")

    def test_structural_failure_blocks_export_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("synthetic_br_profiles_gan.services.generation_service._export_dataset") as export_dataset:
                with patch("synthetic_br_profiles_gan.services.generation_service.validate_profile_dataframe") as validation:
                    validation.return_value = SimpleNamespace(
                        report={
                            "is_valid": False,
                            "invalid_rows": 1,
                            "valid_rows": 2,
                            "n_rows": 3,
                            "reason_counts": {"cpf_invalido": 1},
                            "details": {"missing_columns": []},
                        }
                    )
                    with self.assertRaises(StructuralValidationError):
                        run_generation(
                            GenerationRequest(
                                "programmatic",
                                None,
                                3,
                                root / "selected.csv",
                                "csv",
                                seed=41,
                                selected_columns=["Nome"],
                            )
                        )
            export_dataset.assert_not_called()

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

    def test_cli_generate_accepts_columns_and_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected.csv"
            self.assertEqual(
                main(
                    [
                        "--log-level",
                        "ERROR",
                        "generate",
                        "--model",
                        "programmatic",
                        "--rows",
                        "5",
                        "--columns",
                        "Nome,Idade",
                        "Estado",
                        "CPF",
                        "--output",
                        str(selected),
                        "--format",
                        "csv",
                        "--seed",
                        "41",
                    ]
                ),
                0,
            )
            self.assertEqual(list(pd.read_csv(selected, sep=";").columns), ["Nome", "Idade", "Estado", "CPF"])

            preset = root / "preset.csv"
            self.assertEqual(
                main(
                    [
                        "--log-level",
                        "ERROR",
                        "generate",
                        "--model",
                        "programmatic",
                        "--rows",
                        "5",
                        "--preset",
                        "minimo",
                        "--output",
                        str(preset),
                        "--format",
                        "csv",
                        "--seed",
                        "41",
                    ]
                ),
                0,
            )
            self.assertEqual(list(pd.read_csv(preset, sep=";").columns), ["Nome", "Idade", "Estado", "CPF"])


if __name__ == "__main__":
    unittest.main()
