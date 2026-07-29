from __future__ import annotations

import json
import sys
import tempfile
import unittest
import inspect
import tomllib
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
from synthetic_br_profiles_gan.ui.generation_adapter import (
    UIGenerationRequest,
    artifact_status_label,
    artifact_status_warning,
    list_available_artifacts,
    list_generation_artifacts,
    run_ui_generation,
)
from synthetic_br_profiles_gan.ui.model_catalog import model_catalog, model_catalog_by_name
from synthetic_br_profiles_gan.ui.pages import generate as generate_page
from synthetic_br_profiles_gan.ui.pages import governance as governance_page
from synthetic_br_profiles_gan.ui.pages import models as models_page
from synthetic_br_profiles_gan.ui.streamlit_app import NAV_ITEMS
from synthetic_br_profiles_gan.ui.services.audit_service import read_audit_events, sanitize_event, write_audit_event
from synthetic_br_profiles_gan.ui.services.compliance_service import EVIDENCE_STATUSES, build_compliance_matrix, compliance_summary
from synthetic_br_profiles_gan.ui.services.execution_history import filter_history, history_as_rows, load_history
from synthetic_br_profiles_gan.ui.services.governance_service import build_governance_snapshot, default_generation_model
from synthetic_br_profiles_gan.ui.theme import GOVERNANCE_WARNING, INSTITUTIONAL_DESCRIPTION, PALETTE, assert_palette_contrast, contrast_ratio, ui_css
from synthetic_br_profiles_gan.ui.ui_config import UIConfig, load_ui_config, validate_ui_config


def _ui_config(models_root: Path, sessions_root: Path) -> UIConfig:
    return UIConfig(
        title="Teste",
        subtitle="Subtítulo de teste",
        preview_rows=5,
        models_root=models_root,
        sessions_root=sessions_root,
        artifacts_root=models_root.parent,
        default_rows=10,
        min_rows=1,
        limits={"programmatic": 100, "ctgan": 50, "simple_gan": 20},
        default_model="programmatic",
        default_preset="completo",
        default_format="csv",
        default_seed=41,
        approved_model_artifacts={"ctgan": ("ctgan-valid",), "simple_gan": ("simple-valid",)},
        audit_events_path=models_root.parent / "ui_audit" / "events.jsonl",
        legal_content_last_review="2026-07-28",
        raw={},
    )


def _write_neural_artifact(
    root: Path,
    model: str,
    artifact_name: str = "artifact",
    *,
    approval_status: str = "approved",
    vocabulary_version: int = 2,
    purpose: str = "approved",
    created_at_utc: str = "2026-07-28T00:00:00+00:00",
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
        self.assertEqual(
            ", ".join(by_name["ctgan"].recommended_use_cases[:-1]) + f" e {by_name['ctgan'].recommended_use_cases[-1]}",
            "experimentos tabulares, comparação com baseline programático, pesquisa metodológica e uso de artefatos treinados e avaliados pela equipe responsável",
        )
        self.assertEqual(
            ", ".join(by_name["simple_gan"].recommended_use_cases[:-1]) + f" e {by_name['simple_gan'].recommended_use_cases[-1]}",
            "estudos acadêmicos sobre GANs, comparação de arquiteturas, análise de colapso categórico e experimentos metodológicos controlados",
        )
        for entry in entries:
            self.assertTrue(entry.label)
            self.assertTrue(entry.short_description)
            self.assertTrue(entry.detailed_description)
            self.assertTrue(entry.simple_summary)
            self.assertTrue(entry.technical_summary)
            self.assertTrue(entry.best_for)
            self.assertTrue(entry.limitations)
            self.assertTrue(entry.privacy_considerations)
        self.assertIn("não recomendado", " ".join(by_name["simple_gan"].limitations).lower())

    def test_ui_config_loads_and_validates_defaults(self) -> None:
        config = load_ui_config(ROOT / "configs" / "ui.yaml")
        self.assertGreater(config.preview_rows, 0)
        self.assertIn(config.default_model, {"programmatic", "ctgan", "simple_gan"})
        self.assertIn(config.default_format, {"csv", "json", "parquet"})
        self.assertGreaterEqual(config.default_rows, config.min_rows)
        self.assertLessEqual(config.default_rows, config.limits[config.default_model])
        self.assertTrue(config.models_root)
        self.assertTrue(config.sessions_root)
        self.assertTrue(config.artifacts_root)
        self.assertTrue(config.audit_events_path)
        self.assertEqual(config.legal_content_last_review, "2026-07-28")
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

    def test_generation_artifacts_include_valid_artifacts_sorted_by_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            _write_neural_artifact(models_root, "ctgan", "ctgan-v1", vocabulary_version=1, approval_status="approved", created_at_utc="2026-07-25T00:00:00+00:00")
            _write_neural_artifact(models_root, "ctgan", "ctgan-v2-smoke", approval_status="smoke", purpose="smoke", created_at_utc="2026-07-27T00:00:00+00:00")
            _write_neural_artifact(models_root, "ctgan", "ctgan-v2-candidate", approval_status="candidate", purpose="candidate", created_at_utc="2026-07-28T00:00:00+00:00")
            invalid = models_root / "ctgan-invalid"
            invalid.mkdir(parents=True)
            (invalid / "training_manifest.json").write_text("{}", encoding="utf-8")
            config = _ui_config(models_root, root / "sessions")
            all_artifacts = list_available_artifacts(models_root)
            generation_artifacts = list_generation_artifacts(config)
            self.assertEqual({artifact.artifact_id for artifact in all_artifacts["ctgan"]}, {"ctgan-v1", "ctgan-v2-smoke", "ctgan-v2-candidate"})
            self.assertEqual(
                [artifact.artifact_id for artifact in generation_artifacts["ctgan"]],
                ["ctgan-v2-candidate", "ctgan-v2-smoke", "ctgan-v1"],
            )
            by_id = {artifact.artifact_id: artifact for artifact in generation_artifacts["ctgan"]}
            self.assertEqual(artifact_status_label(by_id["ctgan-v2-candidate"]), "Candidato")
            self.assertIn("avaliação", artifact_status_warning(by_id["ctgan-v2-candidate"]) or "")
            self.assertEqual(artifact_status_label(by_id["ctgan-v2-smoke"]), "Smoke")
            self.assertIn("validação técnica", artifact_status_warning(by_id["ctgan-v2-smoke"]) or "")
            self.assertEqual(artifact_status_label(by_id["ctgan-v1"]), "Legado")
            self.assertIn("versão anterior", artifact_status_warning(by_id["ctgan-v1"]) or "")
            self.assertEqual(default_generation_model(config, all_artifacts["ctgan"]), "programmatic")

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
            self.assertTrue(config.audit_events_path.exists())
            events = read_audit_events(config.audit_events_path)
            self.assertTrue(any(event["event"] == "generation_succeeded" for event in events))
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

    def test_theme_palette_has_wcag_aa_contrast_for_core_pairs(self) -> None:
        assert_palette_contrast()
        self.assertGreaterEqual(contrast_ratio(PALETTE["text"], PALETTE["background"]), 4.5)
        self.assertGreaterEqual(contrast_ratio(PALETTE["primary_dark"], PALETTE["background"]), 4.5)
        self.assertGreaterEqual(contrast_ratio(PALETTE["sidebar_text"], PALETTE["sidebar"]), 4.5)
        css = ui_css()
        self.assertIn("align-items: stretch", css)
        self.assertIn("grid-template-columns: 1fr", css)
        self.assertIn("#0F172A", css)
        self.assertIn("#1E3A8A", css)
        self.assertIn('sbp-governance-card-title', css)
        self.assertNotIn('data-baseweb="input"', css)
        self.assertNotIn('data-baseweb="select"', css)
        self.assertNotIn('[data-testid="stNumberInput"]', css)
        self.assertNotIn('[data-testid="stSelectbox"]', css)
        self.assertNotIn('[data-testid="stMultiSelect"]', css)
        self.assertNotIn('outline: none', css)
        self.assertNotIn('background-color: transparent', css)
        self.assertNotIn('.stButton>button {\n  border-bottom', css)
        self.assertIn("sbp-final-warning", css)

    def test_streamlit_theme_config_uses_native_widget_borders(self) -> None:
        with (ROOT / ".streamlit" / "config.toml").open("rb") as handle:
            config = tomllib.load(handle)
        theme = config["theme"]
        sidebar = theme["sidebar"]
        self.assertEqual(theme["base"], "light")
        self.assertEqual(theme["primaryColor"], "#1E3A8A")
        self.assertEqual(theme["backgroundColor"], "#FFFFFF")
        self.assertEqual(theme["secondaryBackgroundColor"], "#E8EEF7")
        self.assertEqual(theme["textColor"], "#0F172A")
        self.assertEqual(theme["borderColor"], "#64748B")
        self.assertIs(theme["showWidgetBorder"], True)
        self.assertIs(theme["showSidebarBorder"], True)
        self.assertEqual(theme["baseRadius"], "medium")
        self.assertEqual(theme["buttonRadius"], "medium")
        self.assertEqual(sidebar["backgroundColor"], "#0F172A")
        self.assertEqual(sidebar["secondaryBackgroundColor"], "#1E293B")
        self.assertEqual(sidebar["borderColor"], "#334155")
        self.assertIs(sidebar["showWidgetBorder"], True)

    def test_navigation_contains_only_three_pages_with_generation_first(self) -> None:
        labels = [label for label, _ in NAV_ITEMS]
        self.assertEqual(labels, ["Gerar dados", "Modelos", "Governança"])
        self.assertNotIn("Visão geral", labels)
        self.assertNotIn("Conformidade regulatória", labels)

    def test_generation_warning_downloads_and_final_callout_visual_contract(self) -> None:
        source = inspect.getsource(generate_page.render_generation_page)
        self.assertIn("<strong>Atenção:</strong>", source)
        self.assertNotIn("! Atenção", source)
        self.assertIn("Os dados gerados são sintéticos", GOVERNANCE_WARNING)
        result_source = inspect.getsource(generate_page._render_generation_result)
        self.assertIn("st.columns([1, 1, 4])", result_source)
        self.assertIn("Baixar dataset", result_source)
        self.assertIn("Baixar manifesto", result_source)

    def test_models_page_is_simplified_without_artifact_history(self) -> None:
        source = inspect.getsource(models_page.render_models_page) + inspect.getsource(models_page._render_model_section)
        self.assertIn("Resumo Simples", inspect.getsource(models_page.render_equal_summary_cards) if hasattr(models_page, "render_equal_summary_cards") else "")
        self.assertIn("Resumo Técnico", inspect.getsource(models_page.render_equal_summary_cards) if hasattr(models_page, "render_equal_summary_cards") else "")
        self.assertNotIn("Histórico de artefatos de modelo", source)
        self.assertNotIn("Ver artefatos disponíveis", source)
        self.assertNotIn("render_flow", source)
        self.assertEqual(
            models_page._indicado_para("ctgan"),
            "Indicado para: Experimentos tabulares, comparação com baseline programático, pesquisa metodológica e uso de artefatos treinados e avaliados pela equipe responsável.",
        )
        self.assertEqual(
            models_page._indicado_para("programmatic"),
            "Indicado para: Desenvolvimento, testes funcionais, demonstrações técnicas, validação de pipelines, ensino e grandes volumes locais.",
        )
        self.assertEqual(
            models_page._indicado_para("simple_gan"),
            "Indicado para: Estudos acadêmicos sobre GANs, comparação de arquiteturas, análise de colapso categórico e experimentos metodológicos controlados.",
        )

    def test_governance_page_keeps_glossary_and_removes_visual_sections(self) -> None:
        source = inspect.getsource(governance_page.render_governance_page)
        glossary_source = inspect.getsource(governance_page._render_glossary)
        self.assertNotIn("Uso responsável e conformidade", source)
        self.assertNotIn("Ela não constitui parecer jurídico", source)
        self.assertNotIn("A matriz abaixo é educacional", source)
        self.assertNotIn("Modelos e versões", source)
        self.assertIn("st.container(border=True)", source)
        self.assertIn("Resumo Operacional", source)
        self.assertIn("Qualidade dos Dados", source)
        self.assertIn("Diversidade e Memorização", source)
        self.assertIn("Realismo Condicional", source)
        self.assertIn("Execuções Recentes", source)
        self.assertIn("Tipo", source)
        self.assertIn("Modelo", source)
        self.assertIn("Status", source)
        self.assertIn("Como interpretar os indicadores", glossary_source)
        self.assertIn("Execuções registradas", glossary_source)
        self.assertIn("Duplicidade de combinações-base", glossary_source)
        self.assertIn("Correspondência exata com holdout", glossary_source)
        self.assertIn("Realismo condicional", glossary_source)
        self.assertIn("Cauda superior", glossary_source)
        self.assertIn("Exact train match rate", glossary_source)
        self.assertIn("Fonte: manifesto de execução", source)

    def test_visual_brand_and_title_case_contract(self) -> None:
        app_source = inspect.getsource(__import__("synthetic_br_profiles_gan.ui.streamlit_app", fromlist=["_render_sidebar_menu"])._render_sidebar_menu)
        catalog = model_catalog_by_name()
        self.assertIn("Dados Sintéticos Brasileiro", app_source)
        self.assertNotIn("Dados Sintéticos BR", app_source)
        self.assertEqual(catalog["ctgan"].label, "CTGAN — Modelo Tabular Avançado")
        self.assertEqual(catalog["simple_gan"].label, "GAN Simples — Experimental")
        self.assertIn("Volume e Reprodutibilidade", inspect.getsource(generate_page.render_generation_page))
        self.assertIn("Usos Recomendados", inspect.getsource(models_page._render_model_section))

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
            directory_result = write_audit_event(root, "session_started", session_id="abc")
            self.assertFalse(directory_result.written)

    def test_governance_history_filters_and_empty_values_do_not_invent_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _ui_config(root / "models", root / "sessions")
            snapshot = build_governance_snapshot(config)
            self.assertEqual(snapshot.overview["latest_run"], "Sem execução registrada")
            self.assertEqual(snapshot.risk_indicators[0]["valor"], "Sem execução registrada")
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
                        },
                        "conditional_income": {
                            "summary": {
                                "status": "diagnóstico",
                                "conditional_groups_compared": 2,
                                "max_conditional_income_wasserstein": 100.0,
                            }
                        },
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

            snapshot = build_governance_snapshot(config)
            self.assertTrue(any(item["indicador"] == "Taxa de duplicidade" for item in snapshot.diversity_memorization_indicators))
            self.assertTrue(any(item["indicador"] == "Maior desvio condicional" for item in snapshot.conditional_realism_indicators))

    def test_compliance_matrix_uses_allowed_statuses_without_certification_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _ui_config(root / "models", root / "sessions")
            snapshot = build_governance_snapshot(config)
            matrix = build_compliance_matrix(config, snapshot)
            statuses = {row["status"] for row in matrix}
            self.assertTrue(statuses.issubset(set(EVIDENCE_STATUSES)))
            summary = compliance_summary(config, snapshot)
            text = json.dumps({"matrix": matrix, "summary": summary}, ensure_ascii=False).lower()
            self.assertNotIn("%", text)
            self.assertNotIn("certificação", text.replace("não constitui parecer jurídico, certificação regulatória", ""))

    def test_institutional_text_uses_updated_sentence(self) -> None:
        self.assertIn(
            "Inteligência Artificial e validação de pipelines de dados",
            INSTITUTIONAL_DESCRIPTION,
        )
        self.assertNotIn("Inteligência Artificial, validação", INSTITUTIONAL_DESCRIPTION)

    def test_streamlit_app_loads_with_apptest_when_available(self) -> None:
        try:
            from streamlit.testing.v1 import AppTest
        except Exception:
            self.skipTest("Streamlit testing is not available.")
        app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"))
        app.run(timeout=20)
        self.assertFalse(app.exception)
        text = "\n".join(element.value for element in app.markdown)
        self.assertIn("Dados Sintéticos Brasileiro", text)
        self.assertNotIn("Dados Sintéticos BR", text)
        self.assertIn("Gerar dados", text)
        self.assertNotIn("Visão geral", text)
        self.assertNotIn("Conformidade regulatória", text)
        self.assertNotIn("Vocabulário v2", text)
        self.assertNotIn("Versão 0.2.0", text)
        self.assertNotIn("Uso responsável e rastreável de dados sintéticos", text)
        self.assertIn("Atenção:", text)
        self.assertNotIn("! Atenção", text)


if __name__ == "__main__":
    unittest.main()
