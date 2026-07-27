from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.benchmark import (
    DEFAULT_BENCHMARK_CONFIG,
    aggregate_summary_by_model,
    benchmark_matrix,
    build_benchmark_id,
    resolve_benchmark_config,
    run_benchmark,
)
from synthetic_br_profiles_gan.cli import main
from synthetic_br_profiles_gan.config import ConfigurationError, deep_merge, load_yaml_config, save_yaml_config
from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable


def _numeric_metric(value: float) -> dict:
    return {
        "reference": {"mean": value, "median": value, "std": 1.0},
        "synthetic": {"mean": value + 1.0, "median": value + 1.0, "std": 1.2},
        "absolute_mean_diff": 1.0,
        "relative_mean_diff": 0.1,
        "wasserstein_distance": 1.0,
        "wasserstein_distance_normalized": 0.5,
        "ks_statistic": 0.2,
        "median_diff": 1.0,
        "std_diff": 0.2,
    }


def _fake_pipeline_result(model_name: str, config: dict) -> dict:
    run_id = f"run-{model_name}-{config['seed']}"
    root = Path(config["artifacts_root"]) / "runs" / run_id / "approved"
    categorical = {
        column: {
            "total_variation_distance": 0.1,
            "missing_categories": [],
            "unexpected_categories": [],
        }
        for column in ["Genero", "Regiao", "Estado", "Escolaridade", "Estado_Civil", "Ocupacao"]
    }
    evaluation = {
        "against_holdout": {
            "numeric": {
                "Idade": _numeric_metric(40.0),
                "Renda": _numeric_metric(3000.0),
                "Dependentes": _numeric_metric(1.0),
            },
            "categorical": categorical,
            "correlations": {
                "summary": {"mean_abs_difference": 0.05, "max_abs_difference": 0.1},
                "pearson": {"absolute_difference": {"Idade": {"Idade": 0.0, "Renda": 0.1}}},
                "spearman": {"absolute_difference": {"Idade": {"Idade": 0.0, "Renda": 0.2}}},
            },
            "categorical_relationships": {
                "Regiao__Estado": {"total_variation_distance": 0.1, "max_cell_difference": 0.05}
            },
            "grouped_income": {
                "Regiao": {"absolute_difference": {"Sudeste": 100.0, "Sul": 50.0}},
                "Escolaridade": {"absolute_difference": {"Ensino Medio": 80.0}},
                "Ocupacao": {"absolute_difference": {"Tecnico": 60.0}},
                "Faixa_Etaria": {"absolute_difference": {"25-34": 70.0}},
            },
        },
        "privacy": {
            "columns_used": ["Idade", "Genero", "Renda"],
            "excluded_columns": ["CPF", "RG", "CNH", "Titulo_Eleitor", "Telefone", "Nome"],
            "duplicate_row_rate": 0.0,
            "unique_combination_rate": 1.0,
            "exact_train_match_rate": 0.0,
            "exact_holdout_match_rate": 0.0,
            "category_coverage_holdout": {column: 1.0 for column in categorical},
            "nearest_neighbor_train": {
                "distance_to_closest_record": {"mean": 0.3, "min": 0.1, "median": 0.2},
                "nearest_neighbor_distance_ratio": {"mean": 0.8, "median": 0.75},
            },
        },
        "row_counts": {"synthetic": config["generation"]["rows"], "train": 80, "holdout": 20},
    }
    return {
        "run_id": run_id,
        "status": "approved",
        "evaluation": evaluation,
        "validation": {"n_rows": 10, "valid_rows": 10, "invalid_rows": 0, "reason_counts": {}, "is_valid": True},
        "quality_gates": {
            "status": "approved",
            "failures": [],
            "metrics_checked": {
                "invalid_rows": 0,
                "duplicated_identifier": 0,
                "null_required_fields": 0,
                "exact_train_match_rate": 0.0,
                "total_variation_distance": 0.1,
                "correlation_difference": 0.05,
            },
        },
        "generation": {"postprocessing_seconds": 0.01},
        "manifest": {"duration_seconds": 1.0, "requested_rows": config["generation"]["rows"], "generated_rows": config["generation"]["rows"]},
        "stage_durations": {
            "training_seconds": 0.1,
            "generation_seconds": 0.2,
            "validation_seconds": 0.01,
            "evaluation_seconds": 0.03,
            "export_seconds": 0.01,
        },
        "paths": {
            "manifest": root / "manifest.json",
            "root_manifest": root.parent / "manifest.json",
            "dataset_parquet": root / "dataset.parquet",
            "evaluation": root / "evaluation.json",
            "quality_gates": root / "quality_gates.json",
        },
    }


class BenchmarkTest(unittest.TestCase):
    def small_config(self, root: Path) -> dict:
        return deep_merge(
            DEFAULT_BENCHMARK_CONFIG,
            {
                "benchmark": {
                    "name": "unit",
                    "models": ["programmatic", "ctgan"],
                    "seeds": [11, 22],
                    "calibration_rows": 120,
                    "synthetic_rows": 20,
                    "holdout_fraction": 0.2,
                    "assessment_mode": "smoke",
                    "continue_on_error": True,
                },
                "generation": {"batch_size": 32, "max_batches": 2},
                "evaluation": {"privacy": {"max_nearest_neighbor_rows": 20}},
                "outputs": {"base_directory": str(root / "benchmarks"), "export_individual_xlsx": False},
            },
        )

    def test_config_validation_matrix_and_benchmark_id(self) -> None:
        config = resolve_benchmark_config(load_yaml_config(ROOT / "configs" / "benchmark.yaml"))
        self.assertEqual(len(benchmark_matrix(config)), 9)
        self.assertRegex(build_benchmark_id("Pilot Benchmark"), r"^pilot-benchmark-\d{8}T\d{6}Z-[0-9a-f]{8}$")

    def test_unknown_model_is_rejected(self) -> None:
        config = deep_merge(DEFAULT_BENCHMARK_CONFIG, {"benchmark": {"models": ["unknown"]}})
        with self.assertRaises(ConfigurationError):
            resolve_benchmark_config(config)

    def test_run_benchmark_reuses_same_calibration_per_seed_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen: dict[int, set[tuple[int, int]]] = {}

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata):
                seen.setdefault(int(config["seed"]), set()).add((id(train), id(holdout)))
                return _fake_pipeline_result(model_name, config)

            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=fake_run_pipeline_on_splits,
            ):
                result = run_benchmark(self.small_config(root))

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_runs"], 4)
            self.assertTrue(result["paths"]["results_parquet"].exists())
            self.assertTrue(result["paths"]["results_csv"].exists())
            self.assertTrue(result["paths"]["summary_json"].exists())
            self.assertTrue(result["paths"]["benchmark_manifest"].exists())
            self.assertTrue(all(len(ids) == 1 for ids in seen.values()))
            self.assertTrue((result["benchmark_dir"] / "runs" / "programmatic" / "seed-11" / "run-reference.json").exists())

    def test_continue_on_error_true_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata):
                if model_name == "ctgan":
                    raise RuntimeError("ctgan failed")
                return _fake_pipeline_result(model_name, config)

            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=fake_run_pipeline_on_splits,
            ):
                result = run_benchmark(self.small_config(root))

            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["completed_runs"], 2)
            self.assertEqual(result["failed_runs"], 2)
            self.assertTrue(result["paths"]["failures_json"].exists())
            self.assertEqual({failure["stage"] for failure in result["failures"]}, {"run"})

    def test_continue_on_error_false_stops_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.small_config(Path(tmp))
            config["benchmark"]["continue_on_error"] = False
            config["benchmark"]["models"] = ["ctgan"]

            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=RuntimeError("stop"),
            ):
                with self.assertRaises(RuntimeError):
                    run_benchmark(config)

    def test_optional_dependency_absence_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.small_config(Path(tmp))
            config["benchmark"]["models"] = ["ctgan"]
            with patch(
                "synthetic_br_profiles_gan.benchmark._preflight_models",
                return_value=[{"model": "ctgan", "message": "Install with: pip install -e \".[ctgan]\""}],
            ):
                result = run_benchmark(config)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_runs"], 2)
            self.assertIn("Install with", result["failures"][0]["message"])

    def test_aggregate_summary_by_model(self) -> None:
        rows = [
            {"model": "programmatic", "status": "approved", "renda_ks": 0.1, "training_seconds": 0.0},
            {"model": "programmatic", "status": "quarantined", "renda_ks": 0.3, "training_seconds": 0.0},
        ]
        aggregate = aggregate_summary_by_model(rows, [], ["programmatic"])
        self.assertEqual(aggregate["programmatic"]["completed_runs"], 2)
        self.assertEqual(aggregate["programmatic"]["approved"], 1)
        self.assertAlmostEqual(aggregate["programmatic"]["metrics"]["renda_ks"]["mean"], 0.2)
        self.assertIn("ci95_exploratory", aggregate["programmatic"]["metrics"]["renda_ks"])

    def test_cli_benchmark_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "benchmark.yaml"
            save_yaml_config(self.small_config(root), config_path)

            with patch("synthetic_br_profiles_gan.cli.run_benchmark") as fake:
                fake.return_value = {
                    "benchmark_id": "bench-1",
                    "status": "completed",
                    "benchmark_dir": root,
                    "completed_runs": 1,
                    "failed_runs": 0,
                }
                exit_code = main(
                    [
                        "--log-level",
                        "ERROR",
                        "benchmark",
                        "--config",
                        str(config_path),
                        "--models",
                        "programmatic",
                        "--seeds",
                        "11",
                    ]
                )
            self.assertEqual(exit_code, 0)
            called_config = fake.call_args.args[0]
            self.assertEqual(called_config["benchmark"]["models"], ["programmatic"])
            self.assertEqual(called_config["benchmark"]["seeds"], [11])

    @unittest.skipUnless(os.environ.get("RUN_SLOW_MODEL_TESTS") == "1", "slow optional benchmark test")
    def test_real_three_model_smoke_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = deep_merge(
                DEFAULT_BENCHMARK_CONFIG,
                {
                    "benchmark": {
                        "name": "slow-smoke",
                        "models": ["programmatic", "simple_gan", "ctgan"],
                        "seeds": [11],
                        "calibration_rows": 80,
                        "synthetic_rows": 20,
                        "assessment_mode": "smoke",
                    },
                    "models": {
                        "simple_gan": {"epochs": 1, "batch_size": 16, "latent_dim": 8, "verbose_every": 0, "metrics_every": 0},
                        "ctgan": {"epochs": 1, "batch_size": 20, "enable_gpu": False, "verbose": False, "cuda": None},
                    },
                    "generation": {"batch_size": 24, "max_batches": 2},
                    "evaluation": {"privacy": {"max_nearest_neighbor_rows": 20}},
                    "outputs": {"base_directory": str(Path(tmp) / "benchmarks"), "export_individual_xlsx": False},
                },
            )
            result = run_benchmark(config)
            self.assertIn(result["status"], {"completed", "completed_with_failures"})
            self.assertGreaterEqual(result["completed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
