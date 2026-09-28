"""Testes rápidos da API HTTP usada pelo frontend React."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

try:
    from fastapi.testclient import TestClient
    from synthetic_br_profiles_gan.api.app import create_app
except Exception:  # pragma: no cover - permite rodar a suíte sem o extra opcional api.
    TestClient = None
    create_app = None

from synthetic_br_profiles_gan.api.settings import ApiSettings
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS, MODEL_COLUMNS
from synthetic_br_profiles_gan.services.generation_service import GenerationResult


@unittest.skipIf(TestClient is None, "FastAPI não está instalado.")
class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = ApiSettings(
            models_root=root / "models",
            web_sessions_root=root / "web_sessions",
            artifacts_root=root / "artifacts",
            audit_events_path=root / "audit" / "events.jsonl",
            cors_origins=("http://localhost:5173",),
            row_limits={"programmatic": 100, "ctgan": 50, "simple_gan": 20},
        )
        self.client = TestClient(create_app(settings=self.settings))
        self.session_id = str(uuid4())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_health_does_not_expose_local_paths(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertNotIn("models_root", json.dumps(payload))
        self.assertNotIn("web_sessions_root", json.dumps(payload))

    def test_generation_requires_ephemeral_session_header(self) -> None:
        response = self.client.post(
            "/api/generations",
            json={"model": "programmatic", "num_rows": 2, "output_format": "csv", "seed": 41},
        )

        self.assertEqual(response.status_code, 400)

    def test_generation_rejects_internal_paths_in_payload(self) -> None:
        response = self.client.post(
            "/api/generations",
            headers={"X-UI-Session-ID": self.session_id},
            json={
                "model": "programmatic",
                "num_rows": 2,
                "output_format": "csv",
                "seed": 41,
                "output_path": "artifacts/leak.csv",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_programmatic_generation_job_is_scoped_by_session(self) -> None:
        with patch("synthetic_br_profiles_gan.api.jobs.run_generation", side_effect=_fake_run_generation):
            response = self.client.post(
                "/api/generations",
                headers={"X-UI-Session-ID": self.session_id},
                json={
                    "model": "programmatic",
                    "num_rows": 2,
                    "output_format": "csv",
                    "seed": 41,
                    "selected_columns": ["Nome", "Idade"],
                },
            )

        self.assertEqual(response.status_code, 202)
        generation_id = response.json()["generation_id"]
        payload = self._wait_for_completion(generation_id)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["exported_columns"], ["Nome", "Idade"])

        preview = self.client.get(
            f"/api/generations/{generation_id}/preview",
            headers={"X-UI-Session-ID": self.session_id},
        )
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json()["columns"], ["Nome", "Idade"])

        wrong_session = self.client.get(
            f"/api/generations/{generation_id}",
            headers={"X-UI-Session-ID": str(uuid4())},
        )
        self.assertEqual(wrong_session.status_code, 404)

    def test_train_endpoint_is_not_exposed(self) -> None:
        response = self.client.post("/api/train")

        self.assertEqual(response.status_code, 404)

    def test_generation_failure_status_does_not_expose_traceback(self) -> None:
        def fail_generation(request):
            raise RuntimeError("erro interno privado")

        with patch("synthetic_br_profiles_gan.api.jobs.run_generation", side_effect=fail_generation):
            response = self.client.post(
                "/api/generations",
                headers={"X-UI-Session-ID": self.session_id},
                json={"model": "programmatic", "num_rows": 2, "output_format": "csv", "seed": 41},
            )

        self.assertEqual(response.status_code, 202)
        generation_id = response.json()["generation_id"]
        payload = self._wait_for_completion(generation_id)
        self.assertEqual(payload["status"], "failed")
        dumped = json.dumps(payload, ensure_ascii=False).lower()
        self.assertIn("runtimeerror", dumped)
        self.assertNotIn("traceback", dumped)
        self.assertNotIn("erro interno privado", dumped)

    def test_model_artifacts_expose_ids_not_paths(self) -> None:
        artifact_dir = _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved")
        response = self.client.get("/api/models/ctgan/artifacts")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["artifacts"]), 1)
        artifact = payload["artifacts"][0]
        self.assertEqual(artifact["artifact_id"], artifact_dir.relative_to(self.settings.models_root).as_posix())
        self.assertNotIn("artifact_path", artifact)
        self.assertNotIn("\\", artifact["artifact_id"])
        self.assertNotIn("\\", payload["recommended_artifact_id"])
        self.assertTrue(payload["recommended_artifact_id"])

    def test_models_contract_exposes_three_models_and_recommended_ctgan(self) -> None:
        _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved")

        response = self.client.get("/api/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        by_name = {entry["name"]: entry for entry in payload["models"]}
        self.assertEqual(set(by_name), {"programmatic", "ctgan", "simple_gan"})
        self.assertEqual(payload["default_model"], "programmatic")
        self.assertTrue(by_name["programmatic"]["available"])
        self.assertTrue(by_name["programmatic"]["recommended"])
        self.assertEqual(by_name["ctgan"]["recommended_artifact"]["status"], "Aprovado")
        self.assertTrue(by_name["ctgan"]["recommended_artifact"]["recommended"])
        self.assertTrue(by_name["simple_gan"]["experimental"])

    def test_model_detail_and_recommended_artifact_endpoints(self) -> None:
        _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved")

        detail = self.client.get("/api/models/ctgan")
        recommended = self.client.get("/api/models/ctgan/recommended")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(recommended.status_code, 200)
        self.assertEqual(detail.json()["name"], "ctgan")
        artifact = recommended.json()["artifact"]
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact["vocabulary_version"], 2)
        self.assertEqual(artifact["income_model_version"], 3)
        self.assertEqual(artifact["geography_model_version"], 2)
        self.assertIsNone(artifact["epochs"])
        self.assertIsNone(artifact["conditional_income_status"])

    def test_governance_endpoints_are_sanitized_and_preserve_nulls(self) -> None:
        _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved", include_evidence=True)
        self.settings.audit_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.audit_events_path.write_text(
            json.dumps(
                {
                    "event": "generation_completed",
                    "session_id": "abc",
                    "timestamp_utc": "2026-07-30T00:00:00Z",
                    "details": {"path": "C:/secret/file.csv", "cpf": "123"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        for path in [
            "/api/governance",
            "/api/governance/summary",
            "/api/governance/quality",
            "/api/governance/privacy",
            "/api/governance/income",
            "/api/governance/executions",
            "/api/governance/audit",
        ]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            dumped = json.dumps(response.json(), ensure_ascii=False).lower()
            self.assertNotIn("artifact_path", dumped)
            self.assertNotIn("traceback", dumped)
            self.assertNotIn("c:/secret", dumped)
            self.assertNotIn("hostname", dumped)
            self.assertNotIn("username", dumped)

        payload = self.client.get("/api/governance").json()
        self.assertEqual(payload["governance_decision"]["evaluation_status"], "approved")
        self.assertEqual(payload["governance_decision"]["recommendation_status"], "recommended")
        self.assertFalse(payload["governance_decision"]["general_default"])
        self.assertEqual(payload["governance_decision"]["production_status"], "not_approved")
        self.assertEqual(payload["governance_decision"]["mandatory_passed"], 3)
        self.assertEqual(payload["governance_decision"]["mandatory_total"], 3)
        self.assertEqual(len(payload["available_strategies"]), 3)
        strategies = {item["model"]: item for item in payload["available_strategies"]}
        self.assertEqual(strategies["programmatic"]["role"], "Padrão geral da plataforma")
        self.assertEqual(strategies["ctgan"]["recommended_artifact_id"], "ctgan/approved")
        self.assertEqual(strategies["simple_gan"]["role"], "Baseline acadêmico experimental")
        self.assertTrue(payload["quality_gates"])
        self.assertTrue(all("source" in gate for gate in payload["quality_gates"]))
        self.assertTrue(any(gate["mandatory"] for gate in payload["quality_gates"]))
        self.assertTrue(any(item["label"] == "Benchmark" for item in payload["provenance"]["items"]))
        self.assertEqual(payload["recommended_model"]["metrics"]["duplicate_base_row_rate"], 0.0)
        self.assertIsNone(payload["recommended_model"]["metrics"]["conditional_income_status"])
        self.assertTrue(any(item["term"] == "Geo_Key" for item in payload["glossary"]))

    def test_governance_reports_missing_artifact_without_breaking(self) -> None:
        response = self.client.get("/api/governance")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["recommended_model"])
        self.assertEqual(len(payload["available_strategies"]), 3)
        self.assertEqual(payload["governance_decision"]["evaluation_status"], "not_evaluated")
        self.assertFalse(payload["governance_decision"]["general_default"])
        self.assertEqual(payload["provenance"]["items"], [])
        values = [item["value"] for item in payload["operational"]["metrics"]]
        self.assertIn(None, values)

    def test_governance_evidence_is_available_per_model(self) -> None:
        _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved", include_evidence=True)
        _write_fake_governance_run(self.settings.artifacts_root, "programmatic", "approved", 0.00005, 0.059)
        _write_fake_governance_run(self.settings.artifacts_root, "simple_gan", "quarantined", None, 0.984)

        response = self.client.get("/api/governance")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        evidence = payload["evidence_by_model"]
        self.assertEqual(evidence["default_model"], "ctgan")
        by_model = {item["model"]: item for item in evidence["models"]}
        self.assertEqual(set(by_model), {"programmatic", "ctgan", "simple_gan"})
        self.assertEqual(by_model["programmatic"]["source_identifier"], "run-programmatic")
        self.assertEqual(by_model["programmatic"]["privacy"]["diversity_memorization"][6]["value"], 0.00005)
        self.assertEqual(by_model["ctgan"]["artifact_id"], "ctgan/approved")
        self.assertTrue(by_model["ctgan"]["quality_gates"])
        self.assertIn("Experimental", by_model["simple_gan"]["status_label"])
        self.assertEqual(by_model["simple_gan"]["latest_execution_status"], "quarantined")
        self.assertIsNone(by_model["simple_gan"]["privacy"]["diversity_memorization"][6]["value"])
        dumped = json.dumps(payload, ensure_ascii=False).lower()
        self.assertNotIn(str(self.settings.artifacts_root).lower(), dumped)

    def _wait_for_completion(self, generation_id: str) -> dict[str, object]:
        for _ in range(30):
            response = self.client.get(
                f"/api/generations/{generation_id}",
                headers={"X-UI-Session-ID": self.session_id},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail("Generation job did not finish.")


def _fake_run_generation(request) -> GenerationResult:
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_path.write_text("Nome;Idade\nAna;30\nBruno;41\n", encoding="utf-8")
    manifest_path = request.output_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "artifact_type": "synthetic_dataset",
        "model": request.model or "programmatic",
        "model_artifact": None,
        "output_file": str(request.output_path),
        "source_training_manifest": None,
        "environment": {"python": "test"},
        "git_commit": "abc",
        "rows": request.num_rows,
        "exported_columns": ["Nome", "Idade"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return GenerationResult(
        model=request.model or "programmatic",
        num_rows=request.num_rows,
        output_path=request.output_path,
        manifest_path=manifest_path,
        duration_seconds=0.01,
        validation_report={"is_valid": True},
        internal_columns=tuple(FINAL_COLUMNS),
        exported_columns=("Nome", "Idade"),
    )


def _write_fake_governance_run(
    artifacts_root: Path,
    model: str,
    status: str,
    exact_train_match_rate: float | None,
    total_variation_distance: float | None,
) -> Path:
    run_id = f"run-{model}"
    root = artifacts_root / "runs" / run_id
    status_dir_name = "quarantine" if status == "quarantined" else status
    status_dir = root / status_dir_name
    status_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "model": model,
        "status": status,
        "generated_rows": 20000,
        "timestamp_utc": "2026-07-30T00:00:00Z",
        "seed": 41,
        "categorical_vocabulary_version": 2,
        "income_model_version": 3,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    evaluation = {
        "privacy": {
            "unique_combinations": 19990,
            "unique_combination_rate": 0.9995,
            "duplicate_base_rows": {
                "duplicated_occurrences": 0,
                "duplicated_groups": 0,
                "duplicate_row_rate": 0.0,
            },
            "exact_matches": {
                "train": {
                    "exact_match_count": None if exact_train_match_rate is None else int(exact_train_match_rate * 20000),
                    "exact_match_rate": exact_train_match_rate,
                },
                "holdout": {
                    "exact_match_count": 0,
                    "exact_match_rate": 0.0,
                },
            },
            "nearest_neighbor_train": {
                "distance_to_closest_record": {"mean": 0.25},
                "nearest_neighbor_distance_ratio": {"mean": 0.7},
            },
        },
        "conditional_income": {
            "summary": {
                "conditional_groups_compared": 12,
                "max_conditional_income_wasserstein": 120.5,
                "max_abs_p95_difference": 80.0,
                "max_abs_p99_difference": None if model == "simple_gan" else 100.0,
                "groups_with_excessive_tail": 0,
                "status": "avaliado",
            }
        },
    }
    quality_gates = {
        "failures": [
            {
                "gate": "total_variation_distance_max",
                "metric": "total_variation_distance",
                "value": total_variation_distance,
                "limit": 0.25,
                "mandatory": False,
                "reason": "threshold_failed",
            }
        ]
        if total_variation_distance is not None and total_variation_distance > 0.25
        else [],
        "metrics_checked": {
            "invalid_rows": 0,
            "duplicated_identifier": 0.0,
            "exact_train_match_rate": exact_train_match_rate,
            "duplicate_base_row_rate": 0.0,
            "total_variation_distance": total_variation_distance,
        },
        "status": status,
    }
    (status_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False), encoding="utf-8")
    (status_dir / "quality_gates.json").write_text(json.dumps(quality_gates, ensure_ascii=False), encoding="utf-8")
    return root


def _write_fake_ctgan_artifact(path: Path, *, include_evidence: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "artifact_type": "trained_synthesizer",
        "model": "ctgan",
        "created_at_utc": "2026-07-30T12:32:08Z",
        "seed": 47,
        "training_required": True,
        "train_rows": 20000,
        "model_columns": MODEL_COLUMNS,
        "final_columns": FINAL_COLUMNS,
        "purpose": "approved",
        "approval_status": "approved",
        "recommended_for_neural_generation": True,
        "general_platform_default": False,
        "categorical_vocabulary_version": 2,
        "income_model_version": 3,
        "geography_model_version": 2,
        "geography_catalog_checksum": "checksum",
        "environment": {
            "python_version": "3.13.0",
            "hostname": "private-host",
            "username": "private-user",
            "cpu_count": 64,
            "memory_gb": 128,
            "library_versions": {"ctgan": "0.12.1"},
        },
    }
    if include_evidence:
        manifest.update(
            {
                "confirmation_benchmark": "ctgan-confirmation",
                "confirmation_seeds": [47, 48, 49],
                "holdout_rows": 5000,
                "calibration_rows": 25000,
                "model_size_bytes": 3072000,
                "ctgan_config": {"epochs": 20, "batch_size": 500, "pac": 10, "generator_lr": 0.0002, "discriminator_lr": 0.0002},
                "approval_evidence_summary": {
                    "approved_runs": 3,
                    "by_seed": {
                        "47": {
                            "raw_geographic_validity_rate": 1.0,
                            "raw_global_validity_rate": 0.92,
                            "known_geography_key_rate": 1.0,
                            "state_coverage": 1.0,
                            "municipality_coverage": 1.0,
                            "ddd_coverage": 1.0,
                            "geography_key_coverage": 1.0,
                            "duplicate_base_row_rate": 0.0,
                            "exact_train_match_rate": 0.0,
                            "duplicated_identifiers": 0,
                            "invalid_rows": 0,
                        }
                    },
                },
                "aggregate_metrics": {
                    "approved_confirmation_seeds": 3,
                    "duplicate_base_row_rate_max": 0.0,
                    "exact_train_match_rate_max": 0.0,
                    "known_geography_key_rate_min": 1.0,
                    "geography_key_coverage_min": 1.0,
                },
            }
        )
    (path / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if include_evidence:
        approval_manifest = {
            "artifact_id": "ctgan/approved",
            "previous_status": "recommended_candidate",
            "new_status": "approved",
            "approved_at_utc": "2026-07-30T12:32:08Z",
            "decision_type": "internal_technical_approval",
            "evidence_benchmarks": ["artifacts/benchmarks/ctgan-confirmation"],
            "confirmation_seeds": [47, 48, 49],
            "mandatory_checks": {
                "zero_invalid_rows": {"passed": True, "value": {"invalid_rows": 0, "path": "C:/secret/validation.json"}},
                "zero_duplicate_base": {"passed": True, "value": {"duplicate_base_row_rate": 0.0, "duplicate_base_duplicated_occurrences": 0}},
                "zero_exact_train_match": {"passed": True, "value": {"exact_train_match_rate": 0.0, "exact_train_match_count": 0}},
            },
            "known_limitations": ["Diretor ausente na seed 48."],
            "recommended_for_neural_generation": True,
            "general_platform_default": False,
            "approval_note": "Decisão técnica interna; não constitui certificação externa.",
        }
        (path / "approval_manifest.json").write_text(json.dumps(approval_manifest, ensure_ascii=False), encoding="utf-8")
    for name in ["model.pkl", "metadata.json", "metadata_ctgan.json"]:
        (path / name).write_text("{}", encoding="utf-8")
    return path
