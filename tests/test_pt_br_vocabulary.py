from __future__ import annotations

import inspect
import json
import random
import sys
import tempfile
import unicodedata
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.calibration import (
    EDUCATION_INCOME_MULTIPLIER,
    _sample_income,
    generate_calibration_dataset,
)
from synthetic_br_profiles_gan.domain.brazil import STATE_MUNICIPALITIES
from synthetic_br_profiles_gan.domain.occupations import (
    ALL_EDUCATION_LEVELS,
    OCCUPATION_CATALOG,
    eligible_occupation_profiles,
    income_multiplier_for_occupation,
    is_occupation_compatible,
    occupation_sampling_weights,
)
from synthetic_br_profiles_gan.generators.demographics import criar_faker, finalizar_perfis_sinteticos
from synthetic_br_profiles_gan.localization import (
    CATEGORICAL_VOCABULARY_VERSION,
    DATA_LOCALE,
    INCOME_MODEL_VERSION,
    LEGACY_CATEGORY_ALIASES,
    UNICODE_NORMALIZATION,
    normalize_text_value,
)
from synthetic_br_profiles_gan.metadata import (
    EDUCATION_CATEGORIES,
    FINAL_COLUMNS,
    MARITAL_STATUS_CATEGORIES,
    OCCUPATION_CATEGORIES,
    default_metadata,
)
from synthetic_br_profiles_gan.models.registry import list_saved_model_artifacts
from synthetic_br_profiles_gan.services.generation_service import (
    _export_dataset,
    build_generation_manifest,
)
from synthetic_br_profiles_gan.services.training_service import TrainingRequest, run_training
from synthetic_br_profiles_gan.column_catalog import resolve_column_selection
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


LEGACY_VALUES = {
    "Ensino Medio",
    "Pos-graduacao",
    "Uniao Estavel",
    "Viuvo",
    "Servicos Gerais",
    "Tecnico",
    "Autonomo",
}


class PtBRVocabularyTest(unittest.TestCase):
    def test_canonical_categories_are_accented_and_nfc(self) -> None:
        self.assertEqual(
            EDUCATION_CATEGORIES,
            ["Fundamental", "Ensino Médio", "Superior Incompleto", "Superior Completo", "Pós-graduação"],
        )
        self.assertEqual(
            MARITAL_STATUS_CATEGORIES,
            ["Solteiro", "Casado", "União Estável", "Divorciado", "Viúvo"],
        )
        for value in [*EDUCATION_CATEGORIES, *MARITAL_STATUS_CATEGORIES, *OCCUPATION_CATEGORIES]:
            self.assertEqual(unicodedata.normalize("NFC", value), value)
        self.assertEqual(DATA_LOCALE, "pt-BR")
        self.assertEqual(UNICODE_NORMALIZATION, "NFC")
        self.assertEqual(CATEGORICAL_VOCABULARY_VERSION, 2)
        self.assertEqual(INCOME_MODEL_VERSION, 2)

    def test_municipalities_are_accented_and_nfc(self) -> None:
        expected = {
            "Maceió",
            "Palmeira dos Índios",
            "Macapá",
            "Vitória da Conquista",
            "Brasília",
            "Ceilândia",
            "Vitória",
            "Goiânia",
            "Anápolis",
            "Aparecida de Goiânia",
            "São Luís",
            "Cuiabá",
            "Várzea Grande",
            "Rondonópolis",
            "Belém",
            "Santarém",
            "João Pessoa",
            "Maringá",
            "Niterói",
            "Petrópolis",
            "Mossoró",
            "Ji-Paraná",
            "Rorainópolis",
            "Caracaraí",
            "Florianópolis",
            "Blumenau",
            "São Paulo",
            "Araguaína",
        }
        municipalities = {city for cities in STATE_MUNICIPALITIES.values() for city in cities}
        self.assertTrue(expected.issubset(municipalities))
        for city in municipalities:
            self.assertEqual(unicodedata.normalize("NFC", city), city)

    def test_programmatic_generation_has_no_legacy_categories(self) -> None:
        frame = generate_calibration_dataset(1000, seed=41)
        values = set()
        for column in ["Escolaridade", "Estado_Civil", "Ocupacao", "Municipio"]:
            values.update(str(value) for value in frame[column].dropna().unique())
        self.assertTrue(LEGACY_VALUES.isdisjoint(values))
        self.assertGreaterEqual(frame["Ocupacao"].nunique(), 20)
        for value in values:
            self.assertEqual(unicodedata.normalize("NFC", value), value)

    def test_occupation_catalog_is_structured_and_reachable(self) -> None:
        self.assertGreaterEqual(len(OCCUPATION_CATALOG), 30)
        names = [profile.name for profile in OCCUPATION_CATALOG]
        self.assertEqual(len(names), len(set(names)))
        reached: set[str] = set()
        for profile in OCCUPATION_CATALOG:
            self.assertTrue(profile.description)
            self.assertGreater(profile.income_multiplier, 0)
            self.assertGreater(profile.sampling_weight, 0)
            self.assertTrue(set(profile.allowed_education).issubset(set(ALL_EDUCATION_LEVELS)))
            self.assertGreaterEqual(profile.minimum_age, 18)
            if profile.maximum_age is not None:
                self.assertGreaterEqual(profile.maximum_age, profile.minimum_age)
        for education in ALL_EDUCATION_LEVELS:
            for age in [18, 21, 24, 30, 45, 65]:
                reached.update(profile.name for profile in eligible_occupation_profiles(education, age))
        self.assertEqual(set(names), reached)

    def test_higher_education_professions_are_restricted(self) -> None:
        restricted = {
            "Médico",
            "Dentista",
            "Engenheiro",
            "Arquiteto",
            "Advogado",
            "Enfermeiro",
            "Professor",
            "Contador",
        }
        for occupation in restricted:
            self.assertFalse(is_occupation_compatible(occupation, "Fundamental", 40))
            self.assertFalse(is_occupation_compatible(occupation, "Ensino Médio", 40))
            self.assertTrue(is_occupation_compatible(occupation, "Superior Completo", 40))
            self.assertTrue(is_occupation_compatible(occupation, "Pós-graduação", 40))

    def test_age_weights_for_student_intern_and_retiree(self) -> None:
        young_weights = occupation_sampling_weights("Superior Incompleto", 22)
        older_weights = occupation_sampling_weights("Superior Incompleto", 34)
        self.assertGreater(young_weights["Estagiário"], older_weights["Estagiário"])
        self.assertGreater(young_weights["Estudante"], older_weights["Estudante"])

        mature_weights = occupation_sampling_weights("Fundamental", 65)
        early_weights = occupation_sampling_weights("Fundamental", 50)
        self.assertGreater(mature_weights["Aposentado"], early_weights["Aposentado"])

    def test_income_trends_are_aggregate_and_not_gender_based(self) -> None:
        self.assertNotIn("genero", inspect.signature(_sample_income).parameters)
        self.assertNotIn("gender", inspect.signature(_sample_income).parameters)

        def incomes(occupation: str, education: str, seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return np.asarray(
                [
                    _sample_income(
                        rng,
                        age=42,
                        education=education,
                        occupation=occupation,
                        region="Sudeste",
                        minimum=800.0,
                        maximum=50000.0,
                    )
                    for _ in range(800)
                ]
            )

        medico = incomes("Médico", "Superior Completo", 1)
        atendente = incomes("Atendente", "Ensino Médio", 2)
        engenheiro = incomes("Engenheiro", "Superior Completo", 3)
        servicos = incomes("Serviços Gerais", "Fundamental", 4)
        gerente = incomes("Gerente", "Superior Completo", 5)
        auxiliar = incomes("Auxiliar Administrativo", "Ensino Médio", 6)

        self.assertGreater(medico.mean(), atendente.mean())
        self.assertGreater(engenheiro.mean(), servicos.mean())
        self.assertGreater(gerente.mean(), auxiliar.mean())
        self.assertLess(medico.min(), atendente.max())
        for sample in [medico, atendente, engenheiro, servicos, gerente, auxiliar]:
            self.assertGreaterEqual(sample.min(), 800.0)
            self.assertLessEqual(sample.max(), 50000.0)

        first = incomes("Analista de Dados", "Pós-graduação", 99)
        second = incomes("Analista de Dados", "Pós-graduação", 99)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(income_multiplier_for_occupation("Médico"), income_multiplier_for_occupation("Atendente"))
        self.assertGreater(EDUCATION_INCOME_MULTIPLIER["Pós-graduação"], EDUCATION_INCOME_MULTIPLIER["Fundamental"])

    def test_mechanic_high_income_is_possible_but_rare(self) -> None:
        rng = np.random.default_rng(2026)
        sample = np.asarray(
            [
                _sample_income(
                    rng,
                    age=48,
                    education="Ensino Médio",
                    occupation="Mecânico",
                    region="Sudeste",
                    minimum=800.0,
                    maximum=50000.0,
                )
                for _ in range(3000)
            ]
        )
        high_income_rate = float((sample > 10000.0).mean())
        self.assertGreater(sample.max(), 10000.0)
        self.assertLess(high_income_rate, 0.03)
        self.assertLess(float(np.median(sample)), 6000.0)
        self.assertGreater(float(np.quantile(sample, 0.99)), float(np.quantile(sample, 0.95)))

    def test_legacy_aliases_are_applied_before_validation(self) -> None:
        for old, new in LEGACY_CATEGORY_ALIASES.items():
            self.assertEqual(normalize_text_value(old), new)

        core = pd.DataFrame(
            [
                {
                    "Idade": 30,
                    "Genero": "Feminino",
                    "Regiao": "Sudeste",
                    "Estado": "SP",
                    "Municipio": "Sao Paulo",
                    "Escolaridade": "Ensino Medio",
                    "Estado_Civil": "Uniao Estavel",
                    "Ocupacao": "Tecnico",
                    "Renda": 2500.0,
                    "Dependentes": 1,
                    "DDD": 11,
                }
            ]
        )
        final = finalizar_perfis_sinteticos(
            core,
            fake=criar_faker(41),
            referencia=pd.Timestamp("2026-07-26").to_pydatetime(),
            rng=random.Random(41),
        )
        row = final.iloc[0]
        self.assertEqual(row["Municipio"], "São Paulo")
        self.assertEqual(row["Escolaridade"], "Ensino Médio")
        self.assertEqual(row["Estado_Civil"], "União Estável")
        self.assertEqual(row["Ocupacao"], "Técnico")
        self.assertTrue(validate_profile_dataframe(final, reference_date="2026-07-26").report["is_valid"])

    def test_exports_preserve_unicode_in_csv_json_and_parquet(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Escolaridade": "Pós-graduação",
                    "Estado_Civil": "União Estável",
                    "Ocupacao": "Médico",
                    "Municipio": "João Pessoa",
                },
                {
                    "Escolaridade": "Ensino Médio",
                    "Estado_Civil": "Solteiro",
                    "Ocupacao": "Técnico",
                    "Municipio": "São Paulo",
                },
                {
                    "Escolaridade": "Superior Completo",
                    "Estado_Civil": "Casado",
                    "Ocupacao": "Analista de Dados",
                    "Municipio": "Belém",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "dados.csv"
            json_path = root / "dados.json"
            parquet_path = root / "dados.parquet"
            _export_dataset(frame, csv_path, "csv")
            _export_dataset(frame, json_path, "json")
            _export_dataset(frame, parquet_path, "parquet")

            self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            pd.testing.assert_frame_equal(pd.read_csv(csv_path, sep=";"), frame)
            pd.testing.assert_frame_equal(pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8"))), frame)
            pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)

    def test_manifests_and_registry_record_vocabulary_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = run_training(TrainingRequest("programmatic", root / "model", {}, seed=41, train_rows=12))
            manifest = json.loads(trained.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["data_locale"], "pt-BR")
            self.assertEqual(manifest["unicode_normalization"], "NFC")
            self.assertEqual(manifest["categorical_vocabulary_version"], 2)
            self.assertEqual(manifest["income_model_version"], 2)

            output = root / "dataset.csv"
            output.write_text("x", encoding="utf-8")
            selection = resolve_column_selection(None, available_columns=FINAL_COLUMNS)
            generated = build_generation_manifest(
                model="ctgan",
                model_artifact=str(root / "legacy"),
                rows=1,
                columns=FINAL_COLUMNS,
                internal_columns=FINAL_COLUMNS,
                column_selection=selection,
                output_format="csv",
                seed=41,
                output_path=output,
                validation_report={"is_valid": True},
                timings={"total_seconds": 0.01},
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                training_manifest={"created_at_utc": "2026-07-28T00:00:00+00:00"},
                candidate_validation={},
                generation_accounting={},
            )
            self.assertEqual(generated["source_model_vocabulary_version"], 1)
            self.assertEqual(generated["output_vocabulary_version"], 2)
            self.assertEqual(generated["source_model_income_version"], 1)
            self.assertEqual(generated["output_income_model_version"], 2)
            self.assertTrue(generated["legacy_value_normalization_applied"])

            legacy = root / "legacy-programmatic"
            legacy.mkdir()
            (legacy / "config.json").write_text("{}", encoding="utf-8")
            default_metadata().save(legacy / "metadata.json")
            legacy_manifest = {
                "schema_version": 1,
                "artifact_type": "trained_synthesizer",
                "model": "programmatic",
                "model_columns": default_metadata().model_columns,
                "final_columns": default_metadata().final_columns,
                "training_required": False,
            }
            (legacy / "training_manifest.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")
            artifacts = list_saved_model_artifacts(root, model="programmatic")
            by_id = {artifact.artifact_id: artifact for artifact in artifacts}
            self.assertTrue(by_id["legacy-programmatic"].is_legacy_vocabulary)
            self.assertTrue(by_id["legacy-programmatic"].compatibility_normalization_required)
            self.assertEqual(by_id["legacy-programmatic"].categorical_vocabulary_version, 1)
            self.assertEqual(by_id["legacy-programmatic"].income_model_version, 1)
            self.assertTrue(by_id["legacy-programmatic"].is_legacy_income_model)


if __name__ == "__main__":
    unittest.main()
