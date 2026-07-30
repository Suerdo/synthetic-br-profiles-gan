from __future__ import annotations

import sys
import tempfile
import unittest
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.config import deep_merge
from synthetic_br_profiles_gan.manifest import build_manifest, build_run_id, get_git_commit, hash_file, write_json
from synthetic_br_profiles_gan.pipeline import DEFAULT_PIPELINE_CONFIG, run_pipeline


class ManifestArtifactsTest(unittest.TestCase):
    def test_manifest_hashes_completed_artifacts_and_git_outside_repo_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_json({"ok": True}, root / "artifact.json")
            started = datetime.now(timezone.utc)
            manifest = build_manifest(
                run_id=build_run_id(),
                model="programmatic",
                seed=1,
                requested_rows=1,
                generated_rows=1,
                status="approved",
                config={"seed": 1},
                artifact_paths={"artifact": artifact},
                started_at_utc=started,
                ended_at_utc=datetime.now(timezone.utc),
                root=root,
            )
            self.assertEqual(manifest["artifact_hashes"]["artifact"], hash_file(artifact))
            self.assertIsNone(get_git_commit(root))

    def test_two_pipeline_runs_do_not_overwrite_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deep_merge(
                DEFAULT_PIPELINE_CONFIG,
                {
                    "artifacts_root": str(root / "artifacts"),
                    "seed": 11,
                    "reference_date": "2026-07-26",
                    "model": "programmatic",
                    "calibration": {"seed": 11, "num_rows": 60, "holdout_fraction": 0.25},
                    "models": {"programmatic": {"seed": 22}},
                    "generation": {"rows": 8, "batch_size": 12, "max_batches": 2, "date_format": "%Y-%m-%d"},
                    "quality_gates": {
                        "assessment_mode": "smoke",
                        "min_evaluation_rows": 1,
                        "total_variation_distance_max": {"value": 1.0, "mandatory": False},
                        "correlation_difference_max": {"value": 1.0, "mandatory": False},
                    },
                    "export": {"xlsx": False, "primary_format": "parquet"},
                },
            )
            first = run_pipeline(config, model_name="programmatic")
            second = run_pipeline(config, model_name="programmatic")
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertTrue(first["paths"]["dataset_parquet"].exists())
            self.assertTrue(second["paths"]["dataset_parquet"].exists())
            self.assertNotEqual(first["paths"]["dataset_parquet"], second["paths"]["dataset_parquet"])

    def test_pipeline_persists_raw_and_final_evaluation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deep_merge(
                DEFAULT_PIPELINE_CONFIG,
                {
                    "artifacts_root": str(root / "artifacts"),
                    "seed": 41,
                    "reference_date": "2026-07-26",
                    "model": "programmatic",
                    "calibration": {"seed": 41, "num_rows": 80, "holdout_fraction": 0.25, "income_model_version": 3},
                    "models": {"programmatic": {"seed": 42}},
                    "generation": {"rows": 10, "batch_size": 12, "max_batches": 2, "date_format": "%Y-%m-%d"},
                    "quality_gates": {
                        "assessment_mode": "smoke",
                        "min_evaluation_rows": 1,
                        "total_variation_distance_max": {"value": 1.0, "mandatory": False},
                        "correlation_difference_max": {"value": 1.0, "mandatory": False},
                    },
                    "evaluation": {"privacy": {"max_nearest_neighbor_rows": 10}, "income_realism": {"minimum_group_rows": 1}},
                    "export": {"xlsx": False, "primary_format": "parquet"},
                },
            )
            result = run_pipeline(config, model_name="programmatic")

            self.assertTrue(result["paths"]["raw_evaluation"].exists())
            self.assertTrue(result["paths"]["final_evaluation"].exists())
            self.assertTrue(result["paths"]["raw_final_comparison"].exists())
            self.assertIn("raw_evaluation", result["manifest"]["artifact_hashes"])
            self.assertIn("final_evaluation", result["manifest"]["artifact_hashes"])
            self.assertIn("raw_final_comparison", result["manifest"]["artifact_hashes"])
            generation = json.loads(result["paths"]["generation"].read_text(encoding="utf-8"))
            self.assertIn("raw_final_comparison", generation)


if __name__ == "__main__":
    unittest.main()
