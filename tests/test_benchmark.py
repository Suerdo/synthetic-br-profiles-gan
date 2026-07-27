from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.benchmark import (
    DEFAULT_BENCHMARK_CONFIG,
    aggregate_summary_by_model_and_size,
    aggregate_summary_by_model,
    benchmark_matrix,
    build_benchmark_id,
    calculate_marginal_gains,
    calculate_scalability_limits,
    holdout_rows_for_train_size,
    resolve_benchmark_config,
    rotate_models_for_seed,
    run_benchmark,
)
from synthetic_br_profiles_gan.calibration import generate_calibration_dataset, split_train_holdout
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
    run_id = f"run-{model_name}-{config['seed']}-{config.get('calibration', {}).get('num_rows', 'na')}"
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
        "stage_resources": {
            "memory_before_training_mb": 100.0,
            "memory_after_training_mb": 120.0,
        },
        "resource_monitor": {
            "peak_memory_mb": 130.0,
            "cpu_count": 4,
            "thread_count": 8,
            "psutil_available": True,
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

    def scaling_config(self, root: Path) -> dict:
        config = self.small_config(root)
        config["benchmark"].pop("calibration_rows", None)
        config["benchmark"]["models"] = ["programmatic", "simple_gan", "ctgan"]
        config["benchmark"]["seeds"] = [11, 22, 33]
        config["benchmark"]["train_sizes"] = [1000, 5000, 20000]
        config["benchmark"]["synthetic_rows"] = 20
        config["execution"] = {"parallelism": 1, "warmup_backends": True, "rotate_model_order_by_seed": True}
        config["resource_limits"] = {
            "max_training_seconds_per_run": None,
            "max_total_seconds_per_run": None,
            "max_peak_memory_mb": None,
            "stop_larger_sizes_after_resource_failure": True,
        }
        return config

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

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata, **kwargs):
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

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata, **kwargs):
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

    def test_scaling_train_sizes_validation_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = resolve_benchmark_config(self.scaling_config(Path(tmp)))
        self.assertEqual(holdout_rows_for_train_size(1000, 0.20), 250)
        self.assertEqual(holdout_rows_for_train_size(5000, 0.20), 1250)
        self.assertEqual(holdout_rows_for_train_size(20000, 0.20), 5000)
        matrix = benchmark_matrix(config)
        self.assertEqual(len(matrix), 27)
        self.assertIn({"model": "ctgan", "seed": 33, "train_size": 20000, "holdout_size": 5000, "calibration_rows": 25000}, matrix)

    def test_invalid_scaling_train_sizes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.scaling_config(Path(tmp))
            config["benchmark"]["train_sizes"] = []
            with self.assertRaises(ConfigurationError):
                resolve_benchmark_config(config)
            config["benchmark"]["train_sizes"] = [1000, 1000]
            with self.assertRaises(ConfigurationError):
                resolve_benchmark_config(config)
            config["benchmark"]["train_sizes"] = [0]
            with self.assertRaises(ConfigurationError):
                resolve_benchmark_config(config)

    def test_split_accepts_exact_train_and_holdout_rows(self) -> None:
        calibration = generate_calibration_dataset(config={"seed": 11, "num_rows": 1250})
        train, holdout = split_train_holdout(calibration, holdout_fraction=0.2, seed=11, train_rows=1000, holdout_rows=250)
        self.assertEqual(train.shape[0], 1000)
        self.assertEqual(holdout.shape[0], 250)

    def test_split_exact_sizes_for_scaling_matrix(self) -> None:
        for train_size, holdout_size in [(1000, 250), (5000, 1250), (20000, 5000)]:
            calibration = pd.DataFrame({"row": range(train_size + holdout_size)})
            train, holdout = split_train_holdout(
                calibration,
                holdout_fraction=0.2,
                seed=11,
                train_rows=train_size,
                holdout_rows=holdout_size,
            )
            self.assertEqual(train.shape[0], train_size)
            self.assertEqual(holdout.shape[0], holdout_size)

    def test_scaling_reuses_same_split_between_models_per_seed_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen: dict[tuple[int, int], set[tuple[int, int]]] = {}

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata, **kwargs):
                key = (int(config["seed"]), int(config["calibration"]["num_rows"]))
                seen.setdefault(key, set()).add((id(train), id(holdout)))
                return _fake_pipeline_result(model_name, config)

            config = self.scaling_config(root)
            config["benchmark"]["seeds"] = [11]
            config["benchmark"]["train_sizes"] = [1000, 5000]
            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark._warmup_backends",
                return_value={"programmatic": 0.0, "simple_gan": 0.01, "ctgan": 0.02},
            ), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=fake_run_pipeline_on_splits,
            ):
                result = run_benchmark(config)
            self.assertEqual(result["expected_runs"], 6)
            self.assertTrue(all(len(ids) == 1 for ids in seen.values()))
            self.assertTrue((result["benchmark_dir"] / "calibration" / "seed-11" / "train-1000" / "train.parquet").exists())
            self.assertTrue((result["benchmark_dir"] / "runs" / "ctgan" / "seed-11" / "train-5000" / "run-reference.json").exists())
            self.assertTrue(result["paths"]["aggregate_by_model_and_size_json"].exists())
            self.assertTrue(result["paths"]["marginal_gains_json"].exists())
            self.assertTrue(result["paths"]["scalability_limits_json"].exists())

    def test_aggregate_by_model_and_size_and_marginal_gains(self) -> None:
        rows = [
            {"model": "ctgan", "seed": 11, "train_size": 1000, "status": "approved", "renda_ks": 0.4, "renda_wasserstein_normalized": 0.8, "training_seconds": 10.0, "peak_memory_mb": 100.0},
            {"model": "ctgan", "seed": 11, "train_size": 5000, "status": "approved", "renda_ks": 0.2, "renda_wasserstein_normalized": 0.5, "training_seconds": 30.0, "peak_memory_mb": 160.0},
            {"model": "ctgan", "seed": 11, "train_size": 20000, "status": "quarantined", "renda_ks": 0.1, "renda_wasserstein_normalized": 0.4, "training_seconds": 90.0, "peak_memory_mb": 300.0},
        ]
        aggregate = aggregate_summary_by_model_and_size(rows, [], ["ctgan"])
        self.assertEqual(aggregate["ctgan"]["1000"]["completed_runs"], 1)
        self.assertAlmostEqual(aggregate["ctgan"]["5000"]["metrics"]["renda_ks"]["mean"], 0.2)
        gains = calculate_marginal_gains(rows)
        comparison = next(row for row in gains if row["comparison"] == "1000_to_5000")
        self.assertAlmostEqual(comparison["renda_wasserstein_normalized_change"], -0.3)
        self.assertGreater(comparison["quality_gain_by_training_second"], 0)

    def test_scalability_limits_distinguish_rejection_from_technical_failure(self) -> None:
        rows = [
            {"model": "simple_gan", "train_size": 1000, "status": "rejected", "resource_limited": False},
            {"model": "simple_gan", "train_size": 5000, "status": "quarantined", "resource_limited": False},
        ]
        failures = [{"model": "simple_gan", "train_size": 20000, "status": "failed"}]
        limits = calculate_scalability_limits(rows, failures, ["simple_gan"], [1000, 5000, 20000])
        self.assertEqual(limits[0]["successful_train_sizes"], [1000, 5000])
        self.assertEqual(limits[0]["first_failed_size"], 20000)

    def test_resource_limit_can_stop_larger_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.scaling_config(root)
            config["benchmark"]["models"] = ["programmatic"]
            config["benchmark"]["seeds"] = [11]
            config["benchmark"]["train_sizes"] = [1000, 5000]
            config["resource_limits"]["max_total_seconds_per_run"] = 0.01

            def fake_run_pipeline_on_splits(config, model_name, train, holdout, metadata, **kwargs):
                result = _fake_pipeline_result(model_name, config)
                result["manifest"]["duration_seconds"] = 1.0
                return result

            with patch("synthetic_br_profiles_gan.benchmark._preflight_models", return_value=[]), patch(
                "synthetic_br_profiles_gan.benchmark.run_pipeline_on_splits",
                side_effect=fake_run_pipeline_on_splits,
            ):
                result = run_benchmark(config)
            self.assertEqual(result["completed_runs"], 1)
            self.assertEqual(result["failed_runs"], 2)
            self.assertTrue(any(failure["stage"] == "resource_limit_skip" for failure in result["failures"]))

    def test_rotation_order_by_seed(self) -> None:
        models = ["programmatic", "simple_gan", "ctgan"]
        self.assertEqual(rotate_models_for_seed(models, 0, True), ["programmatic", "simple_gan", "ctgan"])
        self.assertEqual(rotate_models_for_seed(models, 1, True), ["simple_gan", "ctgan", "programmatic"])
        self.assertEqual(rotate_models_for_seed(models, 2, True), ["ctgan", "programmatic", "simple_gan"])

    def test_cli_benchmark_train_sizes_override(self) -> None:
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
                        "--train-sizes",
                        "1000",
                        "5000",
                    ]
                )
            self.assertEqual(exit_code, 0)
            called_config = fake.call_args.args[0]
            self.assertEqual(called_config["benchmark"]["train_sizes"], [1000, 5000])
            self.assertNotIn("calibration_rows", called_config["benchmark"])

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
