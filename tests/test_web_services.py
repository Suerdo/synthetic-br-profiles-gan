from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.api.settings import load_api_settings
from synthetic_br_profiles_gan.config import ConfigurationError, load_yaml_config
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.registry import get_recommended_artifact, list_saved_model_artifacts
from synthetic_br_profiles_gan.services.audit_service import read_audit_events, sanitize_event, write_audit_event
from synthetic_br_profiles_gan.services.execution_history import filter_history, history_as_rows, load_history
from synthetic_br_profiles_gan.services.governance_service import build_governance_snapshot, default_generation_model
from synthetic_br_profiles_gan.services.model_catalog import model_catalog, model_catalog_by_name
from synthetic_br_profiles_gan.services.training_service import TrainingRequest, run_training


def _write_neural_artifact(
    root: Path,
    model: str,
    artifact_name: str = "artifact",
    *,
    approval_status: str = "approved",
    vocabulary_version: int = 2,
    purpose: str = "approved",
    created_at_utc: str = "2026-07-28T00:00:00+00:00",
    recommended_for_neural_generation: bool = False,
) -> Path:
    metadata = default_metadata()
    artifact = root / artifact_name
    artifact.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "trained_synthesizer",
        "model": model,
        "created_at_utc": created_at_utc,
        "seed": 41,
        "train_rows": 20,
        "training_required": True,
        "model_size_bytes": 123,
        "model_columns": metadata.model_columns,
        "final_columns": metadata.final_columns,
        "data_locale": "pt-BR",
        "unicode_normalization": "NFC",
        "categorical_vocabulary_version": vocabulary_version,
        "approval_status": approval_status,
        "purpose": purpose,
        "recommended_for_neural_generation": recommended_for_neural_generation,
        "general_platform_default": False,
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


class WebServicesTest(unittest.TestCase):
    def test_model_catalog_contains_three_models_with_expected_roles(self) -> None:
        entries = model_catalog()
        self.assertEqual(len(entries), 3)
        by_name = model_catalog_by_name()
        self.assertEqual(set(by_name), {"programmatic", "ctgan", "simple_gan"})
        self.assertTrue(by_name["programmatic"].recommended)
        self.assertFalse(by_name["programmatic"].requires_saved_artifact)
        self.assertEqual(by_name["ctgan"].status_label, "Artefato neural recomendado")
        self.assertTrue(by_name["ctgan"].requires_saved_artifact)
        self.assertTrue(by_name["simple_gan"].experimental)
        self.assertTrue(by_name["simple_gan"].requires_saved_artifact)
        for entry in entries:
            self.assertTrue(entry.label)
            self.assertTrue(entry.short_description)
            self.assertTrue(entry.detailed_description)
            self.assertTrue(entry.simple_summary)
            self.assertTrue(entry.technical_summary)
            self.assertTrue(entry.best_for)
            self.assertTrue(entry.limitations)

    def test_api_config_loads_defaults_and_validates_limits(self) -> None:
        config = load_api_settings(ROOT / "configs" / "api.yaml")
        self.assertGreater(config.preview_rows, 0)
        self.assertIn(config.default_model, {"programmatic", "ctgan", "simple_gan"})
        self.assertIn(config.default_format, {"csv", "json", "parquet"})
        self.assertGreaterEqual(config.default_rows, config.min_rows)
        self.assertLessEqual(config.default_rows, config.row_limits[config.default_model])
        self.assertEqual(config.web_sessions_root.as_posix(), "artifacts/web_sessions")
        self.assertEqual(config.audit_events_path.as_posix(), "artifacts/web_audit/events.jsonl")

        raw = load_yaml_config(ROOT / "configs" / "api.yaml")
        raw["generation"]["limits"]["ctgan"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "api.yaml"
            invalid.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_api_settings(invalid)

    def test_registry_lists_valid_artifacts_and_keeps_programmatic_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            valid = run_training(TrainingRequest("programmatic", models_root / "programmatic-valid", {}, seed=41, train_rows=12))
            _write_neural_artifact(
                models_root,
                "ctgan",
                "ctgan-approved",
                approval_status="approved",
                purpose="approved",
                recommended_for_neural_generation=True,
                created_at_utc="2026-07-30T00:00:00+00:00",
            )
            invalid = models_root / "incomplete"
            invalid.mkdir()
            (invalid / "training_manifest.json").write_text("{}", encoding="utf-8")

            artifacts = list_saved_model_artifacts(models_root)
            ids = {artifact.artifact_id for artifact in artifacts}
            self.assertIn(valid.output_path.relative_to(models_root).as_posix(), ids)
            self.assertIn("ctgan-approved", ids)
            self.assertNotIn("incomplete", ids)

            selected = get_recommended_artifact(models_root, "ctgan")
            self.assertIsNotNone(selected)
            self.assertEqual(selected.artifact_id, "ctgan-approved")
            settings = load_api_settings(ROOT / "configs" / "api.yaml")
            config = type(
                "Config",
                (),
                {
                    "artifacts_root": root,
                    "models_root": models_root,
                    "audit_events_path": root / "audit" / "events.jsonl",
                    "default_model": settings.default_model,
                    "approved_model_artifacts": settings.approved_model_artifacts,
                },
            )()
            self.assertEqual(default_generation_model(config, artifacts), "programmatic")

    def test_audit_events_are_sanitized_and_write_failures_are_tolerated(self) -> None:
        record = sanitize_event(
            "generation_requested",
            {
                "model": "programmatic",
                "rows": 10,
                "CPF": "123.456.789-09",
                "dataset": [{"Nome": "Ana"}],
                "traceback": "stack",
            },
            session_id="session",
        )
        self.assertEqual(record["event"], "generation_requested")
        self.assertEqual(record["model"], "programmatic")
        self.assertNotIn("CPF", record)
        self.assertNotIn("dataset", record)
        self.assertNotIn("traceback", record)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_audit_event(root / "events.jsonl", "session_started", session_id="abc")
            self.assertTrue(result.written)
            self.assertEqual(len(read_audit_events(root / "events.jsonl")), 1)
            directory_result = write_audit_event(root, "session_started", session_id="abc")
            self.assertFalse(directory_result.written)

    def test_governance_history_preserves_missing_values_without_inventing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            config = type(
                "Config",
                (),
                {
                    "artifacts_root": root,
                    "models_root": models_root,
                    "audit_events_path": root / "audit" / "events.jsonl",
                    "default_model": "programmatic",
                    "approved_model_artifacts": {"ctgan": (), "simple_gan": ()},
                },
            )()
            snapshot = build_governance_snapshot(config)
            self.assertEqual(snapshot.overview["latest_run"], "Sem execução registrada")
            self.assertEqual(snapshot.quality_indicators[0]["valor"], "Não avaliado")

            run_dir = root / "runs" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "timestamp_utc": "2026-07-28T00:00:00+00:00",
                        "model": "programmatic",
                        "seed": 41,
                        "status": "approved",
                        "generated_rows": 10,
                        "validation": {"is_valid": True, "reason_counts": {}},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "evaluation.json").write_text(
                json.dumps(
                    {
                        "privacy": {
                            "unique_combinations": 9,
                            "unique_combination_rate": 0.9,
                            "exact_train_match_rate": 0.1,
                            "duplicate_base_rows": {
                                "duplicate_row_rate": 0.1,
                                "duplicated_occurrences": 1,
                                "duplicated_groups": 1,
                            },
                            "exact_matches": {
                                "train": {"exact_match_count": 1, "exact_match_rate": 0.1},
                                "holdout": {"exact_match_count": 0, "exact_match_rate": 0.0},
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            records = load_history(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(filter_history(records, model="programmatic")[0].identifier, "run-a")
            rows = history_as_rows(records)
            self.assertEqual(rows[0]["status"], "approved")
            self.assertEqual(rows[0]["duplicidade_base"], 0.1)
            self.assertEqual(rows[0]["match_exato_treino"], 0.1)
            self.assertEqual(rows[0]["combinações_únicas"], 9)


if __name__ == "__main__":
    unittest.main()
