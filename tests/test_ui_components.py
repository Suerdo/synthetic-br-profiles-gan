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

from synthetic_br_profiles_gan.config import ConfigurationError, load_yaml_config
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.registry import list_saved_model_artifacts
from synthetic_br_profiles_gan.services.generation_service import GenerationResult
from synthetic_br_profiles_gan.services.training_service import TrainingRequest, run_training
from synthetic_br_profiles_gan.ui.generation_adapter import UIGenerationRequest, list_available_artifacts, run_ui_generation
from synthetic_br_profiles_gan.ui.model_catalog import model_catalog, model_catalog_by_name
from synthetic_br_profiles_gan.ui.ui_config import UIConfig, load_ui_config, validate_ui_config


def _ui_config(models_root: Path, sessions_root: Path) -> UIConfig:
    return UIConfig(
        title="Teste",
        preview_rows=5,
        models_root=models_root,
        sessions_root=sessions_root,
        default_rows=10,
        min_rows=1,
        limits={"programmatic": 100, "ctgan": 50, "simple_gan": 20},
        default_model="programmatic",
        default_preset="completo",
        default_format="csv",
        default_seed=41,
        raw={},
    )


def _write_neural_artifact(root: Path, model: str, artifact_name: str = "artifact") -> Path:
    metadata = default_metadata()
    artifact = root / artifact_name
    artifact.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "trained_synthesizer",
        "model": model,
        "created_at_utc": "2026-07-28T00:00:00+00:00",
        "seed": 41,
        "train_rows": 20,
        "training_required": True,
        "model_size_bytes": 123,
        "model_columns": metadata.model_columns,
        "final_columns": metadata.final_columns,
    }
    (artifact / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata.save(artifact / "metadata.json")
    if model == "ctgan":
        (artifact / "model.pkl").write_bytes(b"model")
        (artifact / "metadata_ctgan.json").write_text("{}", encoding="utf-8")
    else:
        for name in ["generator.keras", "discriminator.keras", "preprocessor.pkl", "config.json", "training_history.json"]:
            (artifact / name).write_text("{}", encoding="utf-8")
    return artifact


def _fake_generation_result(output_path: Path) -> GenerationResult:
    manifest_path = output_path.with_suffix(".manifest.json")
    frame = pd.DataFrame({"Nome": ["Ana"], "CPF": ["123.456.789-09"]})
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False, sep=";")
        output_format = "csv"
    elif output_path.suffix == ".json":
        output_path.write_text(json.dumps(frame.to_dict(orient="records")), encoding="utf-8")
        output_format = "json"
    else:
        frame.to_parquet(output_path, index=False)
        output_format = "parquet"
    manifest = {
        "model": "programmatic",
        "rows": 1,
        "format": output_format,
        "seed": 41,
        "exported_columns": ["Nome", "CPF"],
        "output_size_bytes": output_path.stat().st_size,
        "timings": {"total_seconds": 0.01},
        "validation": {"is_valid": True, "reason_counts": {}, "details": {"missing_columns": []}},
        "model_artifact": None,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return GenerationResult(
        model="programmatic",
        num_rows=1,
        output_path=output_path,
        manifest_path=manifest_path,
        duration_seconds=0.01,
        validation_report=manifest["validation"],
        internal_columns=tuple(default_metadata().final_columns),
        exported_columns=("Nome", "CPF"),
    )


class UIComponentsTest(unittest.TestCase):
    def test_model_catalog_contains_three_models_with_expected_flags(self) -> None:
        entries = model_catalog()
        self.assertEqual(len(entries), 3)
        by_name = model_catalog_by_name()
        self.assertTrue(by_name["programmatic"].recommended)
        self.assertFalse(by_name["programmatic"].requires_saved_artifact)
        self.assertEqual(by_name["ctgan"].status_label, "Avançado")
        self.assertTrue(by_name["ctgan"].requires_saved_artifact)
        self.assertTrue(by_name["simple_gan"].experimental)
        self.assertTrue(by_name["simple_gan"].requires_saved_artifact)
        for entry in entries:
            self.assertTrue(entry.label)
            self.assertTrue(entry.short_description)
            self.assertTrue(entry.detailed_description)
            self.assertTrue(entry.best_for)
            self.assertTrue(entry.limitations)

    def test_ui_config_loads_and_validates_defaults(self) -> None:
        config = load_ui_config(ROOT / "configs" / "ui.yaml")
        self.assertGreater(config.preview_rows, 0)
        self.assertIn(config.default_model, {"programmatic", "ctgan", "simple_gan"})
        self.assertIn(config.default_format, {"csv", "json", "parquet"})
        self.assertGreaterEqual(config.default_rows, config.min_rows)
        self.assertLessEqual(config.default_rows, config.limits[config.default_model])
        self.assertTrue(config.models_root)
        raw = load_yaml_config(ROOT / "configs" / "ui.yaml")
        raw["generation"]["limits"]["ctgan"] = 0
        with self.assertRaises(ConfigurationError):
            validate_ui_config(raw)

    def test_registry_lists_only_valid_artifacts_inside_models_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = run_training(TrainingRequest("programmatic", root / "programmatic-valid", {}, seed=41, train_rows=12))
            _write_neural_artifact(root, "ctgan", "ctgan-valid")
            invalid = root / "incomplete"
            invalid.mkdir()
            (invalid / "training_manifest.json").write_text("{}", encoding="utf-8")
            (root / "model.pkl").write_bytes(b"not-a-directory")
            artifacts = list_saved_model_artifacts(root)
            ids = {artifact.artifact_id for artifact in artifacts}
            self.assertIn(valid.output_path.name, ids)
            self.assertIn("ctgan-valid", ids)
            self.assertNotIn("incomplete", ids)
            self.assertEqual([artifact.model for artifact in list_saved_model_artifacts(root, model="ctgan")], ["ctgan"])

    def test_adapter_creates_unique_directory_and_programmatic_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _ui_config(root / "models", root / "sessions")
            captured = []

            def fake_run_generation(request):
                captured.append(request)
                return _fake_generation_result(request.output_path)

            with patch("synthetic_br_profiles_gan.ui.generation_adapter.run_generation", side_effect=fake_run_generation):
                first = run_ui_generation(
                    UIGenerationRequest(
                        model="programmatic",
                        rows=1,
                        output_format="csv",
                        seed=41,
                        config=config,
                        selected_columns=["Nome", "CPF"],
                        session_id="session",
                    )
                )
                second = run_ui_generation(
                    UIGenerationRequest(
                        model="programmatic",
                        rows=1,
                        output_format="json",
                        seed=41,
                        config=config,
                        column_preset="minimo",
                        session_id="session",
                    )
                )
            self.assertNotEqual(first.session_dir, second.session_dir)
            self.assertTrue(first.session_dir.exists())
            self.assertEqual(captured[0].model, "programmatic")
            self.assertIsNone(captured[0].model_path)
            self.assertEqual(captured[0].selected_columns, ["Nome", "CPF"])
            self.assertEqual(captured[1].column_preset, "minimo")
            self.assertEqual(list(first.dataset.columns), ["Nome", "CPF"])
            self.assertTrue(first.manifest_path.exists())

    def test_adapter_uses_saved_artifact_id_without_accepting_arbitrary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            sessions_root = root / "sessions"
            artifact = _write_neural_artifact(models_root, "ctgan", "ctgan-valid")
            config = _ui_config(models_root, sessions_root)
            artifacts = list_available_artifacts(models_root)
            self.assertEqual(artifacts["ctgan"][0].artifact_id, "ctgan-valid")
            captured = []

            def fake_run_generation(request):
                captured.append(request)
                return _fake_generation_result(request.output_path)

            with patch("synthetic_br_profiles_gan.ui.generation_adapter.run_generation", side_effect=fake_run_generation):
                result = run_ui_generation(
                    UIGenerationRequest(
                        model="ctgan",
                        artifact_id="ctgan-valid",
                        rows=1,
                        output_format="parquet",
                        seed=41,
                        config=config,
                        column_preset="minimo",
                    )
                )
            self.assertIsNone(captured[0].model)
            self.assertEqual(captured[0].model_path, artifact.resolve())
            self.assertTrue(result.output_path.exists())
            with self.assertRaises(ConfigurationError):
                run_ui_generation(
                    UIGenerationRequest(
                        model="ctgan",
                        artifact_id="../ctgan-valid",
                        rows=1,
                        output_format="csv",
                        seed=41,
                        config=config,
                        column_preset="minimo",
                    )
                )

    def test_adapter_preserves_domain_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _ui_config(Path(tmp) / "models", Path(tmp) / "sessions")
            with patch(
                "synthetic_br_profiles_gan.ui.generation_adapter.run_generation",
                side_effect=ConfigurationError("falha controlada"),
            ):
                with patch("synthetic_br_profiles_gan.ui.generation_adapter.LOGGER.exception"):
                    with self.assertRaisesRegex(ConfigurationError, "falha controlada"):
                        run_ui_generation(
                            UIGenerationRequest(
                                model="programmatic",
                                rows=1,
                                output_format="csv",
                                seed=41,
                                config=config,
                                column_preset="minimo",
                            )
                        )


if __name__ == "__main__":
    unittest.main()
