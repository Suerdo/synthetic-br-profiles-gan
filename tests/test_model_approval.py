from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.domain.geography import GEOGRAPHY_MODEL_VERSION, GEO_KEY_COLUMN, geography_catalog_checksum
from synthetic_br_profiles_gan.localization import CATEGORICAL_VOCABULARY_VERSION, INCOME_MODEL_VERSION
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.approval import (
    ApprovalValidationError,
    promote_ctgan_geo_v2_artifact,
    validate_ctgan_geo_v2_approval_evidence,
)
from synthetic_br_profiles_gan.models.registry import get_recommended_artifact, list_saved_model_artifacts


class ModelApprovalTest(unittest.TestCase):
    def test_validates_required_evidence_for_ctgan_geo_v2_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_candidate_artifact(root / "models" / "ctgan" / "candidate")
            benchmark = _write_confirmation_benchmark(root / "benchmarks")

            with patch("synthetic_br_profiles_gan.models.approval.load_saved_synthesizer", return_value=_loaded_stub()):
                report = validate_ctgan_geo_v2_approval_evidence(artifact, benchmark)

            self.assertFalse(report["blocked"])
            self.assertTrue(all(check["passed"] for check in report["mandatory_checks"].values()))
            self.assertEqual(report["summary"]["approved_runs"], 3)
            self.assertEqual(report["occupational_coverage"]["48"]["missing_occupations"], ["Diretor"])

    def test_blocks_approval_when_required_evidence_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_candidate_artifact(root / "models" / "ctgan" / "candidate")
            benchmark = _write_confirmation_benchmark(root / "benchmarks", omit_seed=49)

            with patch("synthetic_br_profiles_gan.models.approval.load_saved_synthesizer", return_value=_loaded_stub()):
                with self.assertRaises(ApprovalValidationError) as context:
                    validate_ctgan_geo_v2_approval_evidence(artifact, benchmark)

            self.assertTrue(context.exception.report["blocked"])
            self.assertFalse(context.exception.report["mandatory_checks"]["three_seeds_present"]["passed"])

    def test_promotes_by_copy_and_preserves_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_candidate_artifact(root / "models" / "ctgan" / "candidate")
            benchmark = _write_confirmation_benchmark(root / "benchmarks")
            approved_at = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

            with patch("synthetic_br_profiles_gan.models.approval.load_saved_synthesizer", return_value=_loaded_stub()):
                result = promote_ctgan_geo_v2_artifact(source, benchmark, root / "models" / "ctgan", approved_at_utc=approved_at)

            source_manifest = json.loads((source / "training_manifest.json").read_text(encoding="utf-8"))
            approved_manifest = json.loads(result.training_manifest_path.read_text(encoding="utf-8"))
            approval_manifest = json.loads(result.approval_manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(source_manifest["approval_status"], "recommended_candidate")
            self.assertEqual(approved_manifest["approval_status"], "approved")
            self.assertTrue(approved_manifest["recommended_for_neural_generation"])
            self.assertFalse(approved_manifest["general_platform_default"])
            self.assertEqual(approval_manifest["previous_status"], "recommended_candidate")
            self.assertEqual(approval_manifest["new_status"], "approved")
            self.assertTrue((result.approved_artifact / "manifest.before-approval.json").exists())

    def test_registry_selects_approved_recommended_before_newer_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            _write_candidate_artifact(root / "ctgan" / "newer-candidate", created_at="2026-07-31T00:00:00Z")
            approved = _write_candidate_artifact(root / "ctgan" / "approved", created_at="2026-07-30T00:00:00Z")
            manifest_path = approved / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["purpose"] = "approved"
            manifest["approval_status"] = "approved"
            manifest["recommended_for_neural_generation"] = True
            manifest["general_platform_default"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            selected = get_recommended_artifact(root, "ctgan")
            self.assertIsNotNone(selected)
            self.assertEqual(selected.artifact_path.name, "approved")
            self.assertTrue(selected.recommended_for_neural_generation)
            self.assertFalse(selected.general_platform_default)
            self.assertEqual({artifact.artifact_path.name for artifact in list_saved_model_artifacts(root, "ctgan")}, {"newer-candidate", "approved"})

    def test_public_schema_does_not_expose_geo_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_candidate_artifact(root / "models" / "ctgan" / "candidate")
            benchmark = _write_confirmation_benchmark(root / "benchmarks")

            with patch("synthetic_br_profiles_gan.models.approval.load_saved_synthesizer", return_value=_loaded_stub()):
                report = validate_ctgan_geo_v2_approval_evidence(artifact, benchmark)

            self.assertTrue(report["mandatory_checks"]["geo_key_absent_from_public_schema"]["passed"])
            manifest = json.loads((artifact / "training_manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn(GEO_KEY_COLUMN, manifest["model_columns"])
            self.assertNotIn(GEO_KEY_COLUMN, manifest["final_columns"])


def _write_candidate_artifact(path: Path, *, created_at: str = "2026-07-30T00:00:00Z") -> Path:
    metadata = default_metadata()
    path.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "trained_synthesizer",
        "model": "ctgan",
        "purpose": "recommended_candidate",
        "approval_status": "recommended_candidate",
        "created_at_utc": created_at,
        "seed": 49,
        "training_required": True,
        "train_rows": 20000,
        "holdout_rows": 5000,
        "model_size_bytes": 123,
        "model_columns": metadata.model_columns,
        "final_columns": metadata.final_columns,
        "data_locale": "pt-BR",
        "unicode_normalization": "NFC",
        "categorical_vocabulary_version": CATEGORICAL_VOCABULARY_VERSION,
        "income_model_version": INCOME_MODEL_VERSION,
        "geography_model_version": GEOGRAPHY_MODEL_VERSION,
        "geography_catalog_checksum": geography_catalog_checksum(),
        "geography_catalog_version": 1,
    }
    (path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata.save(path / "metadata.json")
    (path / "model.pkl").write_bytes(b"model")
    (path / "metadata_ctgan.json").write_text("{}", encoding="utf-8")
    return path


def _write_confirmation_benchmark(path: Path, *, omit_seed: int | None = None) -> Path:
    path.mkdir(parents=True)
    seeds = [seed for seed in (47, 48, 49) if seed != omit_seed]
    summary_fields = [
        "model",
        "seed",
        "status",
        "valid_rows",
        "invalid_rows",
        "duplicate_base_row_rate",
        "duplicate_base_duplicated_occurrences",
        "exact_train_match_rate",
        "exact_train_match_count",
        "known_geography_key_rate_raw",
        "raw_geographic_validity_rate",
        "raw_professional_validity_rate",
        "raw_non_relational_validity_rate",
        "state_coverage",
        "municipality_coverage",
        "ddd_coverage",
        "geography_key_coverage_raw",
        "geography_key_distribution_tvd_raw",
        "peak_memory_mb",
        "training_seconds",
        "generation_seconds",
    ]
    with (path / "run_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for seed in seeds:
            writer.writerow(
                {
                    "model": "ctgan",
                    "seed": seed,
                    "status": "approved",
                    "valid_rows": 20000,
                    "invalid_rows": 0,
                    "duplicate_base_row_rate": 0.0,
                    "duplicate_base_duplicated_occurrences": 0.0,
                    "exact_train_match_rate": 0.0,
                    "exact_train_match_count": 0.0,
                    "known_geography_key_rate_raw": 1.0,
                    "raw_geographic_validity_rate": 1.0,
                    "raw_professional_validity_rate": {47: 0.91585, 48: 0.94895, 49: 0.96845}[seed],
                    "raw_non_relational_validity_rate": {47: 0.91555, 48: 0.94835, 49: 0.96845}[seed],
                    "state_coverage": 1.0,
                    "municipality_coverage": 1.0,
                    "ddd_coverage": 1.0,
                    "geography_key_coverage_raw": 1.0,
                    "geography_key_distribution_tvd_raw": 0.1,
                    "peak_memory_mb": 4000.0,
                    "training_seconds": 50.0,
                    "generation_seconds": 11.0,
                }
            )
    result_fields = ["seed", "metric_group", "metric_name", "column", "value", "details"]
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        for seed in seeds:
            coverage = 1.0 if seed in {47, 49} else 36 / 37
            missing = [] if seed in {47, 49} else ["Diretor"]
            rows = [
                (seed, "validation", "duplicated_identifiers", "", 0.0, ""),
                (seed, "quality_gates", "duplicated_identifier", "", 0.0, ""),
                (seed, "quality_gates", "invalid_rows", "", 0.0, ""),
                (seed, "categorical", "category_coverage_holdout", "Ocupacao", coverage, ""),
                (seed, "categorical", "missing_categories_count", "Ocupacao", len(missing), json.dumps(missing)),
            ]
            for row in rows:
                writer.writerow(dict(zip(result_fields, row)))
    (path / "environment.json").write_text(json.dumps({"library_versions": {"ctgan": "0.12.1"}}), encoding="utf-8")
    return path


def _loaded_stub():
    class Synth:
        geography_model_version = GEOGRAPHY_MODEL_VERSION
        geography_catalog_checksum = geography_catalog_checksum()

    class Loaded:
        synthesizer = Synth()

    return Loaded()


if __name__ == "__main__":
    unittest.main()
