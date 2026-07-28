from __future__ import annotations

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

from synthetic_br_profiles_gan.benchmark import DEFAULT_BENCHMARK_CONFIG, resolve_benchmark_config, run_benchmark
from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.config import deep_merge, load_yaml_config
from synthetic_br_profiles_gan.evaluation.vocabulary import evaluate_vocabulary_v2_quality
from synthetic_br_profiles_gan.generators.demographics import criar_faker, finalizar_perfis_sinteticos
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS


def _final_profiles(rows: int = 8, seed: int = 41) -> pd.DataFrame:
    core = generate_calibration_dataset(rows, seed=seed)
    return finalizar_perfis_sinteticos(
        core,
        fake=criar_faker(seed),
        referencia=pd.Timestamp("2026-07-26").to_pydatetime(),
    )


def _fake_pipeline_result(config: dict, model_name: str, **kwargs) -> dict:
    dataset = _final_profiles(int(config["generation"]["rows"]), seed=int(config["seed"]))
    run_id = f"run-{model_name}-{config['seed']}"
    root = Path(config["artifacts_root"]) / "runs" / run_id / "approved"
    categorical = {
        column: {"total_variation_distance": 0.0, "missing_categories": [], "unexpected_categories": []}
        for column in ["Genero", "Regiao", "Estado", "Escolaridade", "Estado_Civil", "Ocupacao"]
    }
    evaluation = {
        "against_holdout": {
            "numeric": {
                "Idade": {"wasserstein_distance": 0.0, "wasserstein_distance_normalized": 0.0, "ks_statistic": 0.0},
                "Renda": {"wasserstein_distance": 0.0, "wasserstein_distance_normalized": 0.0, "ks_statistic": 0.0},
                "Dependentes": {"wasserstein_distance": 0.0, "wasserstein_distance_normalized": 0.0, "ks_statistic": 0.0},
            },
            "categorical": categorical,
            "correlations": {"summary": {"mean_abs_difference": 0.0}},
            "categorical_relationships": {},
            "grouped_income": {},
        },
        "privacy": {
            "duplicate_row_rate": 0.0,
            "unique_combination_rate": 1.0,
            "exact_train_match_rate": 0.0,
            "exact_holdout_match_rate": 0.0,
            "category_coverage_holdout": {column: 1.0 for column in categorical},
            "nearest_neighbor_train": {
                "distance_to_closest_record": {"mean": 0.0},
                "nearest_neighbor_distance_ratio": {"mean": 1.0},
            },
        },
    }
    return {
        "run_id": run_id,
        "status": "approved",
        "dataset": dataset,
        "evaluation": evaluation,
        "validation": {"n_rows": len(dataset), "valid_rows": len(dataset), "invalid_rows": 0, "reason_counts": {}, "is_valid": True},
        "quality_gates": {"status": "approved", "failures": [], "metrics_checked": {"invalid_rows": 0}},
        "generation": {"postprocessing_seconds": 0.01},
        "manifest": {"duration_seconds": 1.0, "requested_rows": len(dataset), "generated_rows": len(dataset)},
        "stage_durations": {"training_seconds": 0.1, "generation_seconds": 0.2, "validation_seconds": 0.01, "evaluation_seconds": 0.01, "export_seconds": 0.01},
        "stage_resources": {},
        "resource_monitor": {},
        "paths": {
            "manifest": root / "manifest.json",
            "root_manifest": root.parent / "manifest.json",
            "dataset_parquet": root / "dataset.parquet",
            "evaluation": root / "evaluation.json",
            "quality_gates": root / "quality_gates.json",
        },
        "model_dir": root / "model",
    }


class VocabularyBenchmarkTest(unittest.TestCase):
    def test_vocabulary_metrics_distinguish_raw_and_final(self) -> None:
        reference = generate_calibration_dataset(200, seed=41)
        final = _final_profiles(30, seed=42)
        raw = reference.head(30).copy()
        raw.loc[0, "Escolaridade"] = "Fundamental"
        raw.loc[0, "Ocupacao"] = "Médico"
        raw.loc[1, "Ocupacao"] = "Tecnico"
        metrics = evaluate_vocabulary_v2_quality(
            reference=reference,
            raw=raw,
            final=final,
            requested_rows=len(final),
            validation_report={"is_valid": True},
        )
        self.assertEqual(metrics["occupation"]["canonical_occupation_count"], 37)
        self.assertIn("Tecnico", metrics["occupation"]["legacy_occupations_raw"])
        self.assertEqual(metrics["occupation"]["legacy_occupations_final"], [])
        self.assertGreater(metrics["coherence"]["education_occupation_invalid_count_raw"], 0)
        self.assertEqual(metrics["coherence"]["education_occupation_invalid_count_final"], 0)
        self.assertTrue(metrics["locale"]["unicode_nfc_valid_final"])
        self.assertEqual(metrics["quality_gates"]["status"], "passed")
        self.assertIn("Gênero não é utilizado", metrics["methodological_notice"])

    def test_vocabulary_metrics_block_final_legacy_value(self) -> None:
        reference = generate_calibration_dataset(40, seed=41)
        final = _final_profiles(10, seed=43)
        final.loc[0, "Ocupacao"] = "Tecnico"
        metrics = evaluate_vocabulary_v2_quality(
            reference=reference,
            raw=reference.head(10),
            final=final,
            requested_rows=len(final),
            validation_report={"is_valid": False},
        )
        self.assertEqual(metrics["quality_gates"]["status"], "failed")
        failure_names = {item["name"] for item in metrics["quality_gates"]["blocking_failures"]}
        self.assertIn("schema_final_valid", failure_names)
        self.assertIn("no_legacy_categories_final", failure_names)

    def test_vocabulary_benchmark_writes_specific_artifacts_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deep_merge(
                DEFAULT_BENCHMARK_CONFIG,
                {
                    "benchmark": {
                        "name": "vocab-unit",
                        "type": "vocabulary_quality",
                        "models": ["programmatic"],
                        "seeds": [41],
                        "train_sizes": [1000],
                        "synthetic_rows": 8,
                        "assessment_mode": "smoke",
                    },
                    "generation": {"batch_size": 16, "max_batches": 2, "date_format": "%Y-%m-%d"},
                    "outputs": {"base_directory": str(root / "benchmarks"), "export_individual_xlsx": False},
                },
            )
            raw = generate_calibration_dataset(8, seed=99)
            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=_fake_pipeline_result,
            ), patch("synthetic_br_profiles_gan.benchmark._sample_raw_synthesizer_output", return_value=raw):
                result = run_benchmark(config)
            self.assertEqual(result["status"], "completed")
            expected = [
                "vocabulary_v2_metrics_csv",
                "occupation_coverage_csv",
                "occupation_distribution_csv",
                "occupation_income_summary_csv",
                "invalid_education_occupation_csv",
                "invalid_age_occupation_csv",
                "rare_occupation_coverage_csv",
                "raw_vs_final_summary_json",
            ]
            for key in expected:
                self.assertIn(key, result["paths"])
                self.assertTrue(result["paths"][key].exists())
            metrics = pd.read_csv(result["paths"]["vocabulary_v2_metrics_csv"])
            self.assertIn("vocabulary_summary", set(metrics["metric_group"]))

    def test_quality_benchmark_config_remains_compatible(self) -> None:
        pilot = resolve_benchmark_config(load_yaml_config(ROOT / "configs" / "benchmark.yaml"))
        self.assertEqual(pilot["benchmark"].get("type"), "quality")
        vocab = resolve_benchmark_config(load_yaml_config(ROOT / "configs" / "benchmark-quality-vocab-v2-smoke.yaml"))
        self.assertEqual(vocab["benchmark"]["type"], "vocabulary_quality")
        self.assertEqual(vocab["benchmark"]["train_sizes"], [1000])
        self.assertEqual(vocab["benchmark"]["models"], ["programmatic", "simple_gan", "ctgan"])


if __name__ == "__main__":
    unittest.main()
