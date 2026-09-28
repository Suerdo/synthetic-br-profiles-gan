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

    def test_model_artifacts_expose_ids_not_paths(self) -> None:
        artifact_dir = _write_fake_ctgan_artifact(self.settings.models_root / "ctgan" / "approved")
        response = self.client.get("/api/models/ctgan/artifacts")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["artifacts"]), 1)
        artifact = payload["artifacts"][0]
        self.assertEqual(artifact["artifact_id"], str(artifact_dir.relative_to(self.settings.models_root)))
        self.assertNotIn("artifact_path", artifact)
        self.assertTrue(payload["recommended_artifact_id"])

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


def _write_fake_ctgan_artifact(path: Path) -> Path:
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
    }
    (path / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    for name in ["model.pkl", "metadata.json", "metadata_ctgan.json"]:
        (path / name).write_text("{}", encoding="utf-8")
    return path
